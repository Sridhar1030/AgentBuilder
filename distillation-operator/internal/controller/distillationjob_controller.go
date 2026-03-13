/*
Copyright 2026.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package controller

import (
	"context"
	"fmt"
	"time"

	corev1 "k8s.io/api/core/v1"
	apierrors "k8s.io/apimachinery/pkg/api/errors"
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
	"k8s.io/apimachinery/pkg/runtime"
	"k8s.io/apimachinery/pkg/types"
	ctrl "sigs.k8s.io/controller-runtime"
	"sigs.k8s.io/controller-runtime/pkg/client"
	logf "sigs.k8s.io/controller-runtime/pkg/log"

	distillationv1alpha1 "github.com/srpillai/distillation-operator/api/v1alpha1"
	"github.com/srpillai/distillation-operator/internal/dsp"
)

const requeueInterval = 30 * time.Second

// DistillationJobReconciler reconciles a DistillationJob object.
type DistillationJobReconciler struct {
	client.Client
	Scheme *runtime.Scheme
}

// +kubebuilder:rbac:groups=distillation.rhoai.example.com,resources=distillationjobs,verbs=get;list;watch;create;update;patch;delete
// +kubebuilder:rbac:groups=distillation.rhoai.example.com,resources=distillationjobs/status,verbs=get;update;patch
// +kubebuilder:rbac:groups=distillation.rhoai.example.com,resources=distillationjobs/finalizers,verbs=update
// +kubebuilder:rbac:groups="",resources=secrets,verbs=get;list;watch

func (r *DistillationJobReconciler) Reconcile(ctx context.Context, req ctrl.Request) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	var job distillationv1alpha1.DistillationJob
	if err := r.Get(ctx, req.NamespacedName, &job); err != nil {
		if apierrors.IsNotFound(err) {
			return ctrl.Result{}, nil
		}
		return ctrl.Result{}, err
	}

	// Initialise phase on first reconcile
	if job.Status.Phase == "" {
		return r.setStatus(ctx, &job, distillationv1alpha1.PhasePending, "", "Initialising")
	}

	switch job.Status.Phase {
	case distillationv1alpha1.PhasePending:
		return r.handlePending(ctx, &job)
	case distillationv1alpha1.PhaseSubmitting:
		return r.handleSubmitting(ctx, &job)
	case distillationv1alpha1.PhaseRunning:
		return r.handleRunning(ctx, &job)
	case distillationv1alpha1.PhaseSucceeded, distillationv1alpha1.PhaseFailed:
		log.Info("DistillationJob in terminal state", "phase", job.Status.Phase)
		return ctrl.Result{}, nil
	}

	return ctrl.Result{}, nil
}

// handlePending resolves the pipeline ID and transitions to Submitting.
func (r *DistillationJobReconciler) handlePending(ctx context.Context, job *distillationv1alpha1.DistillationJob) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	dspClient := dsp.NewClient(job.Namespace)
	pipelineName := job.Spec.PipelineName
	if pipelineName == "" {
		pipelineName = "distillation-flywheel"
	}

	pipelineID, err := dspClient.FindPipelineByName(pipelineName)
	if err != nil {
		log.Error(err, "Failed to find pipeline", "name", pipelineName)
		return r.setStatus(ctx, job, distillationv1alpha1.PhasePending, "",
			fmt.Sprintf("Waiting for pipeline %q: %v", pipelineName, err))
	}

	log.Info("Resolved pipeline", "name", pipelineName, "id", pipelineID)
	job.Status.PipelineRunID = pipelineID // temporarily stash pipeline ID
	return r.setStatus(ctx, job, distillationv1alpha1.PhaseSubmitting, "",
		fmt.Sprintf("Pipeline %q resolved (id=%s), submitting run", pipelineName, pipelineID))
}

// handleSubmitting reads the Groq secret and creates the pipeline run.
func (r *DistillationJobReconciler) handleSubmitting(ctx context.Context, job *distillationv1alpha1.DistillationJob) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	groqKey, err := r.readSecretKey(ctx, job.Namespace, job.Spec.GroqApiKeySecret, "api-key")
	if err != nil {
		log.Error(err, "Failed to read Groq API key secret")
		return r.setStatus(ctx, job, distillationv1alpha1.PhaseFailed, "",
			fmt.Sprintf("Cannot read secret %q: %v", job.Spec.GroqApiKeySecret, err))
	}

	pipelineID := job.Status.PipelineRunID // stashed in Pending phase

	dspClient := dsp.NewClient(job.Namespace)
	displayName := fmt.Sprintf("flywheel-%s", job.Name)

	epochs := job.Spec.Epochs
	if epochs == 0 {
		epochs = 3
	}
	groqModel := job.Spec.GroqModel
	if groqModel == "" {
		groqModel = "llama-3.3-70b-versatile"
	}

	params := dsp.RunParams{
		ModelVersion:     job.Spec.ModelVersion,
		GroqAPIKey:       groqKey,
		GroqModel:        groqModel,
		NumEpochs:        epochs,
		MinGoldThreshold: job.Spec.MinGoldThreshold,
	}

	runID, err := dspClient.CreateRun(displayName, pipelineID, params)
	if err != nil {
		log.Error(err, "Failed to submit pipeline run")
		return r.setStatus(ctx, job, distillationv1alpha1.PhaseFailed, "",
			fmt.Sprintf("Run submission failed: %v", err))
	}

	log.Info("Pipeline run submitted", "runID", runID)
	job.Status.PipelineRunID = runID
	return r.setStatus(ctx, job, distillationv1alpha1.PhaseRunning, runID,
		fmt.Sprintf("Pipeline run %s submitted", runID))
}

// handleRunning polls the run status and transitions to Succeeded or Failed.
func (r *DistillationJobReconciler) handleRunning(ctx context.Context, job *distillationv1alpha1.DistillationJob) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	dspClient := dsp.NewClient(job.Namespace)
	runState, err := dspClient.GetRun(job.Status.PipelineRunID)
	if err != nil {
		log.Error(err, "Failed to poll run status", "runID", job.Status.PipelineRunID)
		return ctrl.Result{RequeueAfter: requeueInterval}, nil
	}

	log.Info("Run status", "runID", job.Status.PipelineRunID, "state", runState.State)

	switch runState.State {
	case "SUCCEEDED":
		return r.setStatus(ctx, job, distillationv1alpha1.PhaseSucceeded, job.Status.PipelineRunID,
			"Pipeline run completed successfully")
	case "FAILED", "SKIPPED", "CANCELED":
		msg := fmt.Sprintf("Pipeline run %s: %s", runState.State, runState.Message)
		return r.setStatus(ctx, job, distillationv1alpha1.PhaseFailed, job.Status.PipelineRunID, msg)
	default:
		// Still running (PENDING, RUNNING, PAUSED, etc.)
		return r.setStatus(ctx, job, distillationv1alpha1.PhaseRunning, job.Status.PipelineRunID,
			fmt.Sprintf("Pipeline run state: %s", runState.State))
	}
}

func (r *DistillationJobReconciler) readSecretKey(ctx context.Context, namespace, secretName, key string) (string, error) {
	var secret corev1.Secret
	if err := r.Get(ctx, types.NamespacedName{Namespace: namespace, Name: secretName}, &secret); err != nil {
		return "", fmt.Errorf("getting secret %s/%s: %w", namespace, secretName, err)
	}
	val, ok := secret.Data[key]
	if !ok {
		return "", fmt.Errorf("key %q not found in secret %s/%s", key, namespace, secretName)
	}
	return string(val), nil
}

func (r *DistillationJobReconciler) setStatus(
	ctx context.Context,
	job *distillationv1alpha1.DistillationJob,
	phase distillationv1alpha1.DistillationJobPhase,
	runID, message string,
) (ctrl.Result, error) {
	log := logf.FromContext(ctx)

	job.Status.Phase = phase
	job.Status.Message = message
	now := metav1.Now()
	job.Status.LastUpdated = &now
	if runID != "" {
		job.Status.PipelineRunID = runID
	}

	if err := r.Status().Update(ctx, job); err != nil {
		log.Error(err, "Failed to update status")
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}

	switch phase {
	case distillationv1alpha1.PhaseSucceeded, distillationv1alpha1.PhaseFailed:
		return ctrl.Result{}, nil
	case distillationv1alpha1.PhaseRunning:
		return ctrl.Result{RequeueAfter: requeueInterval}, nil
	default:
		return ctrl.Result{RequeueAfter: 5 * time.Second}, nil
	}
}

// SetupWithManager sets up the controller with the Manager.
func (r *DistillationJobReconciler) SetupWithManager(mgr ctrl.Manager) error {
	return ctrl.NewControllerManagedBy(mgr).
		For(&distillationv1alpha1.DistillationJob{}).
		Named("distillationjob").
		Complete(r)
}
