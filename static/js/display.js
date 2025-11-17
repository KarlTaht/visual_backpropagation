// Display and rendering functions for the UI

const Display = {
    // Load and display model state
    async loadModelState() {
        try {
            const data = await API.getState();

            if (data.error) {
                document.getElementById('model-info').innerHTML =
                    `<p style="color: red;">Error: ${data.error}</p>`;
                return;
            }

            this.displayModelInfo(data);
            this.displayNetworkVisualization(data);
            Heatmaps.displayColorScale(data);
        } catch (error) {
            document.getElementById('model-info').innerHTML =
                `<p style="color: red;">Failed to load model state: ${error.message}</p>`;
        }
    },

    // Display model info summary
    displayModelInfo(state) {
        const weights = state.weights;
        const activations = state.activations;
        const gradients = state.gradients;

        const html = `
            <div class="info-grid">
                <div class="info-item" title="Intermediate values saved during forward pass. These are needed to compute gradients during backpropagation.">
                    <div class="info-label">Activations Stored</div>
                    <div class="info-value">${Object.keys(activations).length}</div>
                </div>
                <div class="info-item" title="Partial derivatives of the loss with respect to each weight matrix, computed during backward pass via chain rule.">
                    <div class="info-label">Gradients Computed</div>
                    <div class="info-value">${Object.keys(gradients).length}</div>
                </div>
                <div class="info-item" title="Learnable parameter matrices including weights and biases for input, hidden, and output layers.">
                    <div class="info-label">Weight Matrices</div>
                    <div class="info-value">${Object.keys(weights).length}</div>
                </div>
            </div>
        `;

        document.getElementById('model-info').innerHTML = html;
    },

    // Display network visualization with heatmaps
    displayNetworkVisualization(state) {
        const weights = state.weights;
        const activations = state.activations;
        const gradients = state.gradients;

        const inputDim = weights.input_weights[0].length;
        const hiddenDim = weights.hidden_weights[0][0].length;
        const numHiddenLayers = weights.hidden_weights.length;
        const outputDim = weights.output_weights[0].length;

        const minMax = Heatmaps.getGlobalMinMax(state);
        const outputMinMax = Heatmaps.getOutputMinMax(state);
        const gradientMinMax = Heatmaps.getGradientMinMax(state);
        const outputGradientMinMax = Heatmaps.getOutputGradientMinMax(state);

        let html = '';

        // Input Layer
        html += Heatmaps.createLayerRow(
            'Input Layer',
            `(batch, ${inputDim})`,
            'input_weights' in gradients ? `(${inputDim}, ${hiddenDim})` : 'No gradients yet',
            'input_weights' in gradients
        );

        // Hidden Layers
        for (let i = 0; i < numHiddenLayers; i++) {
            const gradKey = `hidden_${i}_weights`;
            html += Heatmaps.createLayerRow(
                `Hidden Layer ${i + 1}`,
                `(batch, ${hiddenDim})`,
                gradKey in gradients ? `(${hiddenDim}, ${hiddenDim})` : 'No gradients yet',
                gradKey in gradients
            );
        }

        // Output Layer
        html += Heatmaps.createLayerRow(
            'Output Layer',
            `(batch, ${outputDim})`,
            'output_weights' in gradients ? `(${hiddenDim}, ${outputDim})` : 'No gradients yet',
            'output_weights' in gradients
        );

        document.getElementById('network-visualization').innerHTML = html;

        // Populate each cell with heatmaps
        Heatmaps.populateLayerCell('Input Layer', weights, activations, gradients, minMax, null, gradientMinMax,
            'input', 'input_weights', 'input_biases', 'hidden_0_pre', 'hidden_0');

        for (let i = 0; i < numHiddenLayers; i++) {
            const hiddenWeights = weights.hidden_weights[i];
            const hiddenBiases = weights.hidden_biases[i];

            Heatmaps.populateLayerCellWithArrays(`Hidden Layer ${i + 1}`, hiddenWeights, hiddenBiases,
                activations, gradients, minMax, null, gradientMinMax,
                `hidden_${i}`, `hidden_${i}_weights`, `hidden_${i}_biases`,
                `hidden_${i+1}_pre`, `hidden_${i+1}`);
        }

        Heatmaps.populateLayerCell('Output Layer', weights, activations, gradients, minMax, outputMinMax, outputGradientMinMax,
            `hidden_${numHiddenLayers}`, 'output_weights', 'output_biases',
            'output', 'output');
    },

    // Load trainer configuration
    async loadTrainerConfig() {
        try {
            const config = await API.getTrainerConfig();

            if (config.error) {
                console.error('Error loading trainer config:', config.error);
                return;
            }

            document.getElementById('learning-rate-input').value = config.learning_rate;
            document.getElementById('current-lr-display').textContent =
                `Current: ${config.learning_rate}`;

            document.getElementById('batch-size-input').value = config.batch_size;
            document.getElementById('current-batch-size-display').textContent =
                `Current: ${config.batch_size}`;

            const lossTypeRadios = document.querySelectorAll('input[name="loss-type"]');
            lossTypeRadios.forEach(radio => {
                radio.checked = (radio.value === config.loss_type);
            });
            document.getElementById('current-loss-type-display').textContent =
                `Current: ${config.loss_type === 'mse' ? 'MSE' : 'Cross-Entropy'}`;

        } catch (error) {
            console.error('Failed to load trainer config:', error);
        }
    },

    // Load training statistics
    async loadTrainingStats() {
        try {
            const data = await API.getTrainingStats();
            await this.displayTrainingStats(data);
        } catch (error) {
            console.error('Failed to load training stats:', error);
            document.getElementById('training-stats').innerHTML =
                `<p style="color: red;">Failed to load training stats: ${error.message}</p>`;
        }
    },

    // Display training statistics
    async displayTrainingStats(stats) {
        const currentLoss = stats.losses.length > 0 ?
            stats.losses[stats.losses.length - 1].toFixed(6) :
            'N/A';

        const minLoss = stats.losses.length > 0 ?
            Math.min(...stats.losses).toFixed(6) :
            'N/A';

        const avgEpochTime = stats.epoch_times && stats.epoch_times.length > 0 ?
            (stats.epoch_times.reduce((a, b) => a + b, 0) / stats.epoch_times.length * 1000).toFixed(1) :
            'N/A';

        const lastEpochTime = stats.epoch_times && stats.epoch_times.length > 0 ?
            (stats.epoch_times[stats.epoch_times.length - 1] * 1000).toFixed(1) :
            'N/A';

        let html = `
            <div class="training-grid">
                <div class="stat-box">
                    <div class="stat-label">Current Epoch</div>
                    <div class="stat-value">${stats.current_epoch}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Current Loss</div>
                    <div class="stat-value">${currentLoss}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Best Loss</div>
                    <div class="stat-value">${minLoss}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Training Steps</div>
                    <div class="stat-value">${stats.losses.length}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Samples / Epoch</div>
                    <div class="stat-value">${stats.samples_per_epoch || 'N/A'}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Total Samples</div>
                    <div class="stat-value">${stats.total_samples_trained || 0}</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Last Epoch Time</div>
                    <div class="stat-value">${lastEpochTime}ms</div>
                </div>
                <div class="stat-box">
                    <div class="stat-label">Avg Epoch Time</div>
                    <div class="stat-value">${avgEpochTime}ms</div>
                </div>
            </div>
        `;

        document.getElementById('training-stats').innerHTML = html;

        // Load multi-run chart
        await this.loadMultiRunChart(stats);
    },

    // Load multi-run chart
    async loadMultiRunChart(currentStats = null) {
        try {
            const runLosses = await API.getRunLosses();

            if (runLosses.error) {
                console.error('Error loading run losses:', runLosses.error);
                if (currentStats && currentStats.losses.length > 0) {
                    Charts.updateSingleRunChart(currentStats);
                }
                return;
            }

            Charts.updateMultiRunChart(runLosses, currentStats);
        } catch (error) {
            console.error('Failed to load multi-run chart:', error);
            if (currentStats && currentStats.losses.length > 0) {
                Charts.updateSingleRunChart(currentStats);
            }
        }
    },

    // Load dataset information
    async loadDatasetInfo() {
        try {
            const data = await API.getDatasetInfo();

            if (data.error) {
                document.getElementById('dataset-info').innerHTML =
                    `<p style="color: #718096;">Dataset information not available</p>`;
                return;
            }

            this.displayDatasetInfo(data);
        } catch (error) {
            document.getElementById('dataset-info').innerHTML =
                `<p style="color: red;">Failed to load dataset info: ${error.message}</p>`;
        }
    },

    // Display dataset information
    displayDatasetInfo(data) {
        let html = '<div class="info-grid">';

        html += `
            <div class="info-item">
                <div class="info-label">Total Samples</div>
                <div class="info-value">${data.num_samples}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Training Sequence Length</div>
                <div class="info-value">${data.sequence_length}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Vocabulary Size</div>
                <div class="info-value">${data.vocab_size}</div>
            </div>
            <div class="info-item">
                <div class="info-label">Task Type</div>
                <div class="info-value" style="font-size: 1em;">${data.task_type}</div>
            </div>
        `;
        html += '</div>';

        // Master sequences section
        if (data.master_sequences) {
            const ms = data.master_sequences;
            html += `
                <div style="margin-top: 25px;">
                    <h3 style="margin-bottom: 15px; color: #2d3748;">Master Sequences</h3>
                    <div class="info-grid">
                        <div class="info-item">
                            <div class="info-label">Number of Master Sequences</div>
                            <div class="info-value">${ms.num_sequences}</div>
                        </div>
            `;

            for (let i = 0; i < ms.sequences.length; i++) {
                const seq = ms.sequences[i].join('');
                const length = ms.sequence_lengths[i];
                html += `
                    <div class="info-item">
                        <div class="info-label">Master Sequence ${i + 1}</div>
                        <div class="info-value" style="font-size: 1.1em; font-family: monospace;">'${seq}'</div>
                        <div style="font-size: 0.9em; color: #718096; margin-top: 5px;">Length: ${length}</div>
                    </div>
                `;
            }

            html += '</div></div>';
        }

        document.getElementById('dataset-info').innerHTML = html;
    },

    // Load current example
    async loadCurrentExample() {
        try {
            const data = await API.getCurrentExample();

            if (data.error) {
                document.getElementById('io-display').style.display = 'none';
                return;
            }

            document.getElementById('io-display').style.display = 'block';

            const datasetData = await API.getDatasetInfo();
            this.displayCurrentExample(data, datasetData);
        } catch (error) {
            console.error('Failed to load current example:', error);
        }
    },

    // Display current example
    displayCurrentExample(example, datasetInfo) {
        this.displayForwardIO(example, datasetInfo);
        this.displayBackwardIO(example, datasetInfo);
    },

    // Display forward pass I/O
    displayForwardIO(example, datasetInfo) {
        const container = document.getElementById('forward-io');

        if (!example.has_dataset) {
            container.innerHTML = '<p style="color: #718096;">Using random data (no dataset)</p>';
            return;
        }

        const input = example.input[0];
        const vocab = datasetInfo.vocabulary;
        const vocabSize = datasetInfo.vocab_size;
        const seqLength = datasetInfo.sequence_length;

        let html = '<div>';

        // Input sequence
        html += '<div style="margin-bottom: 15px;">';
        html += '<div style="font-weight: 600; margin-bottom: 8px; color: #4a5568;">Input Sequence:</div>';
        html += '<div class="sequence-display">';

        const inputLength = seqLength - 1;
        for (let i = 0; i < inputLength; i++) {
            const tokenVec = input.slice(i * vocabSize, (i + 1) * vocabSize);
            const tokenIdx = tokenVec.indexOf(Math.max(...tokenVec));
            const token = vocab[tokenIdx];
            html += `<span class="token">${token}</span>`;
        }
        html += '</div></div>';

        // Target vs prediction
        html += '<div>';
        html += '<div style="font-weight: 600; margin-bottom: 8px; color: #4a5568;">Target vs Prediction:</div>';

        const target = example.target[0];
        const targetIdx = target.indexOf(Math.max(...target));
        const targetToken = vocab[targetIdx];

        const prediction = example.prediction[0];
        const predIdx = prediction.indexOf(Math.max(...prediction));
        const predToken = vocab[predIdx];

        html += '<div class="sequence-display">';
        html += `<span class="token target">${targetToken}</span>`;
        html += '<span style="margin: 0 10px; color: #718096;">vs</span>';
        html += `<span class="token predicted">${predToken}</span>`;
        html += '</div>';

        const isCorrect = targetToken === predToken;
        const statusColor = isCorrect ? '#48bb78' : '#f56565';
        const statusText = isCorrect ? 'CORRECT' : 'INCORRECT';
        html += `<div style="text-align: center; margin-top: 10px; font-weight: 600; color: ${statusColor};">${statusText}</div>`;

        html += '</div></div>';

        container.innerHTML = html;
    },

    // Display backward pass I/O
    displayBackwardIO(example, datasetInfo) {
        const container = document.getElementById('backward-io');

        let html = '<div>';

        const target = example.target[0];
        const prediction = example.prediction[0];
        const vocab = datasetInfo.vocabulary;

        // Target vector
        html += '<div style="margin-bottom: 15px;">';
        html += '<div style="font-weight: 600; margin-bottom: 8px; color: #4a5568;">Target Vector:</div>';
        html += '<div class="vector-display">';
        for (let i = 0; i < vocab.length; i++) {
            html += `<div class="vector-row">`;
            html += `<span class="vector-label">${vocab[i]}:</span>`;
            html += `<span class="vector-value">${target[i].toFixed(4)}</span>`;
            html += `</div>`;
        }
        html += '</div></div>';

        // Prediction vector
        html += '<div style="margin-bottom: 15px;">';
        html += '<div style="font-weight: 600; margin-bottom: 8px; color: #4a5568;">Prediction Vector:</div>';
        html += '<div class="vector-display">';
        for (let i = 0; i < vocab.length; i++) {
            html += `<div class="vector-row">`;
            html += `<span class="vector-label">${vocab[i]}:</span>`;
            html += `<span class="vector-value">${prediction[i].toFixed(4)}</span>`;
            html += `</div>`;
        }
        html += '</div></div>';

        // Loss
        html += '<div class="loss-display">';
        html += '<div class="loss-label">Mean Squared Error Loss</div>';
        html += `<div class="loss-value">${example.loss.toFixed(6)}</div>`;
        html += '</div>';

        html += '</div>';

        container.innerHTML = html;
    },

    // Display master sequence test results
    displayMasterSequenceResults(result) {
        const container = document.getElementById('master-sequence-results');

        let html = `
            <div style="background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                <h3 style="margin: 0 0 10px 0; color: #fff;">Overall Performance</h3>
                <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                    <div class="stat-box">
                        <div class="stat-label">Accuracy</div>
                        <div class="stat-value">${(result.accuracy * 100).toFixed(1)}%</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-label">Correct</div>
                        <div class="stat-value">${result.total_correct} / ${result.total_predictions}</div>
                    </div>
                </div>
            </div>
        `;

        // Each master sequence
        result.master_sequences.forEach((seq, idx) => {
            const seqAccuracy = seq.predictions.filter(p => p.correct).length / seq.predictions.length;

            html += `
                <div style="background: #2a2a2a; padding: 15px; border-radius: 8px; margin-bottom: 15px;">
                    <h4 style="margin: 0 0 10px 0; color: #8b5cf6;">
                        Master Sequence ${idx + 1}: "${seq.master_sequence}"
                        <span style="color: #9ca3af; font-size: 0.9em; font-weight: normal;">
                            (${(seqAccuracy * 100).toFixed(1)}% accuracy)
                        </span>
                    </h4>
                    <div style="max-height: 300px; overflow-y: auto;">
                        <table style="width: 100%; border-collapse: collapse; font-size: 0.9em;">
                            <thead>
                                <tr style="border-bottom: 1px solid #444;">
                                    <th style="padding: 8px; text-align: left; color: #9ca3af;">Input</th>
                                    <th style="padding: 8px; text-align: left; color: #9ca3af;">Target</th>
                                    <th style="padding: 8px; text-align: left; color: #9ca3af;">Predicted</th>
                                    <th style="padding: 8px; text-align: left; color: #9ca3af;">Confidence</th>
                                    <th style="padding: 8px; text-align: center; color: #9ca3af;">Result</th>
                                </tr>
                            </thead>
                            <tbody>
            `;

            seq.predictions.forEach(pred => {
                const resultIcon = pred.correct ? '✓' : '✗';
                const resultColor = pred.correct ? '#10b981' : '#ef4444';

                html += `
                    <tr style="border-bottom: 1px solid #333;">
                        <td style="padding: 8px; font-family: monospace; color: #e0e0e0;">${pred.input}</td>
                        <td style="padding: 8px; font-family: monospace; color: #e0e0e0;">${pred.target}</td>
                        <td style="padding: 8px; font-family: monospace; color: ${pred.correct ? '#10b981' : '#ef4444'};">${pred.predicted}</td>
                        <td style="padding: 8px; color: #e0e0e0;">${(pred.confidence * 100).toFixed(1)}%</td>
                        <td style="padding: 8px; text-align: center; color: ${resultColor}; font-weight: bold;">${resultIcon}</td>
                    </tr>
                `;
            });

            html += `
                            </tbody>
                        </table>
                    </div>
                </div>
            `;
        });

        container.innerHTML = html;
    }
};

window.Display = Display;
