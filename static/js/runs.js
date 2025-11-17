// Training run management functions

const Runs = {
    // Display all runs
    displayRuns(runs, activeRunId) {
        const runListDiv = document.getElementById('run-list');
        const activeRunInfoDiv = document.getElementById('active-run-info');

        if (runs.length === 0) {
            runListDiv.innerHTML = '<p style="color: #9ca3af; margin: 0;">No runs created yet. Click "+ New Run" to start tracking.</p>';
            activeRunInfoDiv.innerHTML = '<span style="color: #9ca3af;">No active run - training data not being saved</span>';
            return;
        }

        // Display active run info
        const activeRun = runs.find(r => r.run_id === activeRunId);
        if (activeRun) {
            const lossDisplay = activeRun.current_loss !== null ? activeRun.current_loss.toFixed(6) : 'N/A';
            const bestLossDisplay = activeRun.best_loss !== null ? activeRun.best_loss.toFixed(6) : 'N/A';
            activeRunInfoDiv.innerHTML = `
                <div style="color: #60a5fa; font-weight: 600; margin-bottom: 5px;">
                    Active: ${activeRun.name}
                </div>
                <div style="color: #9ca3af; font-size: 0.9em;">
                    Epoch: ${activeRun.current_epoch} |
                    Loss: ${lossDisplay} |
                    Best: ${bestLossDisplay} |
                    LR: ${activeRun.config.learning_rate} |
                    Batch: ${activeRun.config.batch_size}
                </div>
            `;
        } else {
            activeRunInfoDiv.innerHTML = '<span style="color: #fbbf24;">No active run selected</span>';
        }

        // Display run list
        let html = '<div style="display: flex; flex-direction: column; gap: 8px;">';

        runs.forEach(run => {
            const isActive = run.run_id === activeRunId;
            const lossDisplay = run.current_loss !== null ? run.current_loss.toFixed(6) : 'N/A';
            const bestLossDisplay = run.best_loss !== null ? run.best_loss.toFixed(6) : 'N/A';
            const bgColor = isActive ? '#1e40af' : '#374151';
            const borderColor = isActive ? '#60a5fa' : '#4b5563';
            const lossTypeDisplay = run.config.loss_type === 'mse' ? 'MSE' : 'CE';

            html += `
                <div style="padding: 10px; background: ${bgColor}; border-radius: 6px; border: 1px solid ${borderColor};">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <div style="display: flex; align-items: center; gap: 8px;">
                            <span id="run-name-${run.run_id}" style="color: #e0e0e0; font-weight: 600;">${run.name}</span>
                            <button onclick="showRenameInput('${run.run_id}', '${run.name.replace(/'/g, "\\'")}')"
                                    style="padding: 2px 6px; font-size: 0.75em; background: #6b7280; border-color: #6b7280;">
                                Rename
                            </button>
                            ${isActive ? '<span style="color: #34d399; font-size: 0.8em;">(ACTIVE)</span>' : ''}
                        </div>
                        <span style="color: #9ca3af; font-size: 0.8em;">${run.run_id}</span>
                    </div>
                    <div id="rename-input-${run.run_id}" style="display: none; margin-bottom: 8px;">
                        <div style="display: flex; gap: 5px;">
                            <input type="text" id="rename-field-${run.run_id}"
                                   style="flex: 1; padding: 4px 8px; background: #1a1a1a; border: 1px solid #444; color: #e0e0e0; border-radius: 4px; font-size: 0.85em;">
                            <button onclick="renameRun('${run.run_id}')"
                                    style="padding: 4px 8px; font-size: 0.85em; background: #059669; border-color: #059669;">Save</button>
                            <button onclick="hideRenameInput('${run.run_id}')"
                                    style="padding: 4px 8px; font-size: 0.85em; background: #6b7280; border-color: #6b7280;">Cancel</button>
                        </div>
                    </div>
                    <div style="color: #9ca3af; font-size: 0.85em; margin-bottom: 4px;">
                        Epochs: ${run.current_epoch} | Loss: ${lossDisplay} | Best: ${bestLossDisplay}
                    </div>
                    <div style="color: #60a5fa; font-size: 0.85em; margin-bottom: 8px;">
                        LR: ${run.config.learning_rate} | Batch: ${run.config.batch_size} | Loss: ${lossTypeDisplay}
                    </div>
                    <div style="display: flex; gap: 5px;">
                        ${!isActive ? `<button onclick="activateRun('${run.run_id}')" style="padding: 4px 10px; font-size: 0.85em; background: #059669; border-color: #059669;">Activate</button>` : ''}
                        <button onclick="resetRun('${run.run_id}')" style="padding: 4px 10px; font-size: 0.85em; background: #d97706; border-color: #d97706;">Reset</button>
                        <button onclick="deleteRun('${run.run_id}')" style="padding: 4px 10px; font-size: 0.85em; background: #dc2626; border-color: #dc2626;">Delete</button>
                    </div>
                </div>
            `;
        });

        html += '</div>';
        runListDiv.innerHTML = html;
    },

    // Show rename input
    showRenameInput(runId, currentName) {
        const inputContainer = document.getElementById(`rename-input-${runId}`);
        const inputField = document.getElementById(`rename-field-${runId}`);
        inputField.value = currentName;
        inputContainer.style.display = 'block';
        inputField.focus();
        inputField.select();
    },

    // Hide rename input
    hideRenameInput(runId) {
        const inputContainer = document.getElementById(`rename-input-${runId}`);
        inputContainer.style.display = 'none';
    },

    // Create a new run
    async createRun() {
        const nameInput = document.getElementById('new-run-name');
        const name = nameInput.value.trim() || null;

        try {
            const result = await API.createRun(name);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Created run: ${result.run.name}`);
            nameInput.value = '';

            await Promise.all([
                this.loadRuns(),
                Display.loadTrainingStats(),
                Display.loadTrainerConfig()
            ]);

        } catch (error) {
            alert(`Failed to create run: ${error.message}`);
        }
    },

    // Load runs from server
    async loadRuns() {
        try {
            const data = await API.getRuns();

            if (data.error) {
                console.error('Error loading runs:', data.error);
                return;
            }

            this.displayRuns(data.runs, data.active_run_id);
        } catch (error) {
            console.error('Failed to load runs:', error);
        }
    },

    // Activate a run
    async activateRun(runId) {
        try {
            const result = await API.activateRun(runId);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Activated run: ${result.run.name}`);

            await Promise.all([
                this.loadRuns(),
                Display.loadTrainingStats(),
                Display.loadTrainerConfig(),
                Display.loadModelState()
            ]);

        } catch (error) {
            alert(`Failed to activate run: ${error.message}`);
        }
    },

    // Reset a run
    async resetRun(runId) {
        if (!confirm('Reset this run to its initial state? This will clear all training history and restore initial weights.')) {
            return;
        }

        try {
            const result = await API.resetRun(runId);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Reset run: ${result.run.name}`);

            await Promise.all([
                this.loadRuns(),
                Display.loadTrainingStats(),
                Display.loadModelState()
            ]);

        } catch (error) {
            alert(`Failed to reset run: ${error.message}`);
        }
    },

    // Rename a run
    async renameRun(runId) {
        const inputField = document.getElementById(`rename-field-${runId}`);
        const newName = inputField.value.trim();

        if (!newName) {
            alert('Name cannot be empty');
            return;
        }

        try {
            const result = await API.renameRun(runId, newName);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Renamed run to: ${result.run.name}`);
            await this.loadRuns();

        } catch (error) {
            alert(`Failed to rename run: ${error.message}`);
        }
    },

    // Delete a run
    async deleteRun(runId) {
        if (!confirm('Delete this run permanently? This cannot be undone.')) {
            return;
        }

        try {
            const result = await API.deleteRun(runId);

            if (result.error) {
                alert(`Error: ${result.error}`);
                return;
            }

            console.log(`Deleted run: ${runId}`);
            await this.loadRuns();

        } catch (error) {
            alert(`Failed to delete run: ${error.message}`);
        }
    }
};

// Make functions available globally for onclick handlers
window.createRun = () => Runs.createRun();
window.loadRuns = () => Runs.loadRuns();
window.activateRun = (id) => Runs.activateRun(id);
window.resetRun = (id) => Runs.resetRun(id);
window.renameRun = (id) => Runs.renameRun(id);
window.deleteRun = (id) => Runs.deleteRun(id);
window.showRenameInput = (id, name) => Runs.showRenameInput(id, name);
window.hideRenameInput = (id) => Runs.hideRenameInput(id);

window.Runs = Runs;
