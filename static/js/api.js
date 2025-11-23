// API communication functions for the gradient visualizer

const API = {
    // Model state
    async getState() {
        const response = await fetch('/api/state');
        return await response.json();
    },

    // Training operations
    async runSingle(updateWeights = false) {
        const response = await fetch('/api/run_single', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ update_weights: updateWeights })
        });
        return await response.json();
    },

    async trainBatch(updateWeights = false) {
        const response = await fetch('/api/train_batch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ update_weights: updateWeights })
        });
        return await response.json();
    },

    async trainEpoch() {
        const response = await fetch('/api/train_epoch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        return await response.json();
    },

    async resetTraining() {
        const response = await fetch('/api/reset_training', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        return await response.json();
    },

    async resetModel() {
        const response = await fetch('/api/reset_model', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        return await response.json();
    },

    // Trainer configuration
    async getTrainerConfig() {
        const response = await fetch('/api/trainer_config');
        return await response.json();
    },

    async updateLearningRate(lr) {
        const response = await fetch('/api/update_learning_rate', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ learning_rate: lr })
        });
        return await response.json();
    },

    async updateBatchSize(batchSize) {
        const response = await fetch('/api/update_batch_size', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ batch_size: batchSize })
        });
        return await response.json();
    },

    async updateLossType(lossType) {
        const response = await fetch('/api/update_loss_type', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ loss_type: lossType })
        });
        return await response.json();
    },

    // Training statistics
    async getTrainingStats() {
        const response = await fetch('/api/training_stats');
        return await response.json();
    },

    // Dataset info
    async getDatasetInfo() {
        const response = await fetch('/api/dataset');
        return await response.json();
    },

    // Current example
    async getCurrentExample() {
        const response = await fetch('/api/current_example');
        return await response.json();
    },

    // Training runs
    async getRuns() {
        const response = await fetch('/api/runs');
        return await response.json();
    },

    async createRun(name = null) {
        const response = await fetch('/api/runs/create', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: name })
        });
        return await response.json();
    },

    async activateRun(runId) {
        const response = await fetch(`/api/runs/${runId}/activate`, {
            method: 'POST'
        });
        return await response.json();
    },

    async resetRun(runId) {
        const response = await fetch(`/api/runs/${runId}/reset`, {
            method: 'POST'
        });
        return await response.json();
    },

    async renameRun(runId, newName) {
        const response = await fetch(`/api/runs/${runId}/rename`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name: newName })
        });
        return await response.json();
    },

    async deleteRun(runId) {
        const response = await fetch(`/api/runs/${runId}`, {
            method: 'DELETE'
        });
        return await response.json();
    },

    async getRunLosses() {
        const response = await fetch('/api/runs/losses');
        return await response.json();
    },

    async getRunGradientNorms() {
        const response = await fetch('/api/runs/gradient_norms');
        return await response.json();
    },

    // Master sequence testing
    async testMasterSequences() {
        const response = await fetch('/api/test_master_sequences');
        return await response.json();
    },

    // Distribution analysis (new feature)
    async getDistributionStats() {
        const response = await fetch('/api/distribution_stats');
        return await response.json();
    }
};

// Export for use in other modules
window.API = API;
