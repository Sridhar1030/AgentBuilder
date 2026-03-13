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

package v1alpha1

import (
	metav1 "k8s.io/apimachinery/pkg/apis/meta/v1"
)

type DistillationJobPhase string

const (
	PhasePending    DistillationJobPhase = "Pending"
	PhaseSubmitting DistillationJobPhase = "Submitting"
	PhaseRunning    DistillationJobPhase = "Running"
	PhaseSucceeded  DistillationJobPhase = "Succeeded"
	PhaseFailed     DistillationJobPhase = "Failed"
)

// DistillationJobSpec defines the desired state of a distillation pipeline run.
type DistillationJobSpec struct {
	// +kubebuilder:default=3
	// +kubebuilder:validation:Minimum=1
	Epochs int `json:"epochs,omitempty"`

	// +optional
	ModelVersion string `json:"modelVersion,omitempty"`

	// +kubebuilder:validation:Required
	GroqApiKeySecret string `json:"groqApiKeySecret"`

	// +kubebuilder:default="llama-3.3-70b-versatile"
	GroqModel string `json:"groqModel,omitempty"`

	// +kubebuilder:default=0
	MinGoldThreshold int `json:"minGoldThreshold,omitempty"`

	// +kubebuilder:default="distillation-flywheel"
	PipelineName string `json:"pipelineName,omitempty"`
}

// DistillationJobStatus defines the observed state of a distillation pipeline run.
type DistillationJobStatus struct {
	// +kubebuilder:validation:Enum=Pending;Submitting;Running;Succeeded;Failed
	Phase DistillationJobPhase `json:"phase,omitempty"`

	PipelineRunID string `json:"pipelineRunID,omitempty"`

	ModelVersion string `json:"modelVersion,omitempty"`

	AvgScore string `json:"avgScore,omitempty"`

	Message string `json:"message,omitempty"`

	LastUpdated *metav1.Time `json:"lastUpdated,omitempty"`
}

// +kubebuilder:object:root=true
// +kubebuilder:subresource:status
// +kubebuilder:printcolumn:name="Phase",type=string,JSONPath=`.status.phase`
// +kubebuilder:printcolumn:name="RunID",type=string,JSONPath=`.status.pipelineRunID`
// +kubebuilder:printcolumn:name="Model",type=string,JSONPath=`.status.modelVersion`
// +kubebuilder:printcolumn:name="Score",type=string,JSONPath=`.status.avgScore`
// +kubebuilder:printcolumn:name="Age",type=date,JSONPath=`.metadata.creationTimestamp`

// DistillationJob is the Schema for the distillationjobs API.
type DistillationJob struct {
	metav1.TypeMeta   `json:",inline"`
	metav1.ObjectMeta `json:"metadata,omitempty"`

	Spec   DistillationJobSpec   `json:"spec,omitempty"`
	Status DistillationJobStatus `json:"status,omitempty"`
}

// +kubebuilder:object:root=true

// DistillationJobList contains a list of DistillationJob.
type DistillationJobList struct {
	metav1.TypeMeta `json:",inline"`
	metav1.ListMeta `json:"metadata,omitempty"`
	Items           []DistillationJob `json:"items"`
}

func init() {
	SchemeBuilder.Register(&DistillationJob{}, &DistillationJobList{})
}
