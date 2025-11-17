// Training controls and UI interaction functions

const Controls = {
    // Get current visualization mode
    getVisualizationMode() {
        const modeRadio = document.querySelector('input[name="viz-mode"]:checked');
        return modeRadio ? modeRadio.value : 'visualize';
    },

    // Run a single training step
    async runSingleStep() {
        try {
            const mode = this.getVisualizationMode();
            const updateWeights = (mode === 'train');

            const result = await API.runSingle(updateWeights);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(result.message);

            // Refresh all displays
            await Promise.all([
                Display.loadModelState(),
                Display.loadCurrentExample(),
                Display.loadTrainingStats()
            ]);

        } catch (error) {
            alert(`Failed to run single step: ${error.message}`);
        }
    },

    // Run a full batch
    async runBatch() {
        try {
            const mode = this.getVisualizationMode();
            const updateWeights = (mode === 'train');

            const result = await API.trainBatch(updateWeights);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            const modeStr = updateWeights ? 'training' : 'visualization';
            console.log(`Batch ${modeStr} completed - Loss: ${result.loss.toFixed(6)}`);

            await Promise.all([
                Display.loadModelState(),
                Display.loadTrainingStats()
            ]);

        } catch (error) {
            alert(`Failed to run batch: ${error.message}`);
        }
    },

    // Run a full epoch
    async runEpoch() {
        try {
            const result = await API.trainEpoch();

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(result.message);

            await Promise.all([
                Display.loadModelState(),
                Display.loadTrainingStats()
            ]);

        } catch (error) {
            alert(`Failed to run epoch: ${error.message}`);
        }
    },

    // Train N epochs
    async trainNEpochs() {
        const input = document.getElementById('epoch-count-input');
        const nEpochs = parseInt(input.value);

        if (!nEpochs || nEpochs < 1) {
            alert('Please enter a valid number of epochs (1 or more)');
            return;
        }

        try {
            for (let i = 0; i < nEpochs; i++) {
                const result = await API.trainEpoch();

                if (result.error) {
                    alert(`Error on epoch ${i + 1}: ${result.error}`);
                    return;
                }

                console.log(`Completed epoch ${i + 1}/${nEpochs} - Loss: ${result.loss.toFixed(6)}`);

                // Update chart after each epoch
                await Display.loadTrainingStats();
            }

            // Refresh model state after all epochs
            await Display.loadModelState();

            console.log(`Training complete! Trained ${nEpochs} epoch(s)`);

        } catch (error) {
            alert(`Failed to train epochs: ${error.message}`);
        }
    },

    // Reset training progress
    async resetTraining() {
        if (!confirm('Reset all training progress?')) {
            return;
        }

        try {
            const result = await API.resetTraining();

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(result.message);
            await Display.loadTrainingStats();

        } catch (error) {
            alert(`Failed to reset training: ${error.message}`);
        }
    },

    // Reset model weights
    async resetModel() {
        if (!confirm('Reset model weights and all training progress? This cannot be undone.')) {
            return;
        }

        try {
            const result = await API.resetModel();

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(result.message);

            await Promise.all([
                Display.loadModelState(),
                Display.loadTrainingStats()
            ]);

        } catch (error) {
            alert(`Failed to reset model: ${error.message}`);
        }
    },

    // Update learning rate
    async updateLearningRate() {
        const input = document.getElementById('learning-rate-input');
        const newLR = parseFloat(input.value);

        if (!newLR || newLR <= 0) {
            alert('Please enter a valid learning rate (greater than 0)');
            return;
        }

        try {
            const result = await API.updateLearningRate(newLR);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Learning rate updated to ${newLR}`);
            alert(`Learning rate updated to ${newLR}`);

            await Display.loadTrainerConfig();

        } catch (error) {
            alert(`Failed to update learning rate: ${error.message}`);
        }
    },

    // Update batch size
    async updateBatchSize() {
        const input = document.getElementById('batch-size-input');
        const newBatchSize = parseInt(input.value);

        if (!newBatchSize || newBatchSize < 1) {
            alert('Please enter a valid batch size (1 or more)');
            return;
        }

        try {
            const result = await API.updateBatchSize(newBatchSize);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Batch size updated to ${newBatchSize}`);
            alert(`Batch size updated to ${newBatchSize}`);

            await Display.loadTrainerConfig();

        } catch (error) {
            alert(`Failed to update batch size: ${error.message}`);
        }
    },

    // Update loss type
    async updateLossType(newLossType) {
        try {
            const result = await API.updateLossType(newLossType);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Loss type updated to ${newLossType}`);
            await Display.loadTrainerConfig();

        } catch (error) {
            alert(`Failed to update loss type: ${error.message}`);
        }
    },

    // Test master sequences
    async testMasterSequences() {
        try {
            const result = await API.testMasterSequences();

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            Display.displayMasterSequenceResults(result);

        } catch (error) {
            alert(`Failed to test master sequences: ${error.message}`);
        }
    }
};

// Make functions available globally for onclick handlers
window.runSingleStep = () => Controls.runSingleStep();
window.runBatch = () => Controls.runBatch();
window.runEpoch = () => Controls.runEpoch();
window.trainNEpochs = () => Controls.trainNEpochs();
window.resetTraining = () => Controls.resetTraining();
window.resetModel = () => Controls.resetModel();
window.updateLearningRate = () => Controls.updateLearningRate();
window.updateBatchSize = () => Controls.updateBatchSize();
window.updateLossType = (type) => Controls.updateLossType(type);
window.testMasterSequences = () => Controls.testMasterSequences();
window.loadModelState = () => Display.loadModelState();
window.loadTrainingStats = () => Display.loadTrainingStats();

window.Controls = Controls;
