// Heatmap and matrix visualization functions

const Heatmaps = {
    // Color conversion for values
    valueToColor(value, min, max, isForward) {
        const absMax = Math.max(Math.abs(min), Math.abs(max));

        if (isForward) {
            // Forward pass: negative (red) -> 0 (white) -> positive (green)
            if (value < 0) {
                const intensity = Math.abs(value) / absMax;
                const r = 255;
                const g = Math.floor(255 - intensity * 255);
                const b = Math.floor(255 - intensity * 255);
                return `rgb(${r}, ${g}, ${b})`;
            } else {
                const intensity = value / absMax;
                const r = Math.floor(255 - intensity * 255);
                const g = 255;
                const b = Math.floor(255 - intensity * 255);
                return `rgb(${r}, ${g}, ${b})`;
            }
        } else {
            // Backward pass: negative (burnt orange) -> 0 (white) -> positive (blue)
            if (value < 0) {
                const intensity = Math.abs(value) / absMax;
                const r = Math.floor(255 - intensity * 50);
                const g = Math.floor(255 - intensity * 115);
                const b = Math.floor(255 - intensity * 215);
                return `rgb(${r}, ${g}, ${b})`;
            } else {
                const intensity = value / absMax;
                const r = Math.floor(255 - intensity * 185);
                const g = Math.floor(255 - intensity * 125);
                const b = 255;
                return `rgb(${r}, ${g}, ${b})`;
            }
        }
    },

    // Render a single heatmap matrix
    renderHeatmap(matrix, containerId, label, minMax, isForward) {
        const container = document.getElementById(containerId);
        if (!container) return '';

        const rows = matrix.length;
        const cols = matrix[0].length;

        let html = `
            <div class="matrix-container">
                <div class="matrix-label">${label} [${rows}×${cols}]</div>
                <div class="matrix-grid" style="grid-template-columns: repeat(${cols}, auto);">
        `;

        for (let i = 0; i < rows; i++) {
            for (let j = 0; j < cols; j++) {
                const value = matrix[i][j];
                const color = this.valueToColor(value, minMax.min, minMax.max, isForward);
                html += `<div class="matrix-cell" style="background-color: ${color};" title="${value.toFixed(4)}"></div>`;
            }
        }

        html += `</div></div>`;
        return html;
    },

    // Average matrix along axis 0 (batch dimension)
    averageAlongAxis0(matrix) {
        if (matrix.length === 0) return [];

        const batchSize = matrix.length;
        const cols = matrix[0].length;
        const result = new Array(cols).fill(0);

        for (let i = 0; i < batchSize; i++) {
            for (let j = 0; j < cols; j++) {
                result[j] += matrix[i][j];
            }
        }

        for (let j = 0; j < cols; j++) {
            result[j] /= batchSize;
        }

        return [result]; // Return as 1×cols matrix
    },

    // Get global min/max for weights and activations (excluding output layer)
    getGlobalMinMax(state) {
        let globalMin = Infinity;
        let globalMax = -Infinity;

        // Check all weights
        const weights = state.weights;
        for (let key in weights) {
            const matrix = weights[key];
            const flat = matrix.flat(Infinity);
            globalMin = Math.min(globalMin, ...flat);
            globalMax = Math.max(globalMax, ...flat);
        }

        // Check all activations EXCEPT output layer activations
        const activations = state.activations;
        for (let key in activations) {
            if (key === 'output' || key.includes('_pre') && key.startsWith('hidden_')) {
                const numHiddenLayers = state.weights.hidden_weights.length;
                const finalPreKey = `hidden_${numHiddenLayers}_pre`;
                if (key === finalPreKey) {
                    continue;
                }
            }

            const matrix = activations[key];
            const flat = matrix.flat(Infinity);
            globalMin = Math.min(globalMin, ...flat);
            globalMax = Math.max(globalMax, ...flat);
        }

        return { min: globalMin, max: globalMax };
    },

    // Get min/max for gradients (excluding output layer)
    getGradientMinMax(state) {
        let gradMin = Infinity;
        let gradMax = -Infinity;

        const gradients = state.gradients;
        for (let key in gradients) {
            if (key === 'output_weights' || key === 'output_biases') {
                continue;
            }
            const matrix = gradients[key];
            const flat = matrix.flat(Infinity);
            gradMin = Math.min(gradMin, ...flat);
            gradMax = Math.max(gradMax, ...flat);
        }

        return { min: gradMin, max: gradMax };
    },

    // Get min/max for output layer gradients only
    getOutputGradientMinMax(state) {
        let gradMin = Infinity;
        let gradMax = -Infinity;

        const gradients = state.gradients;
        if ('output_weights' in gradients) {
            const flat = gradients.output_weights.flat(Infinity);
            gradMin = Math.min(gradMin, ...flat);
            gradMax = Math.max(gradMax, ...flat);
        }
        if ('output_biases' in gradients) {
            const flat = gradients.output_biases.flat(Infinity);
            gradMin = Math.min(gradMin, ...flat);
            gradMax = Math.max(gradMax, ...flat);
        }

        return { min: gradMin, max: gradMax };
    },

    // Get min/max for output layer activations only
    getOutputMinMax(state) {
        let outputMin = Infinity;
        let outputMax = -Infinity;

        const activations = state.activations;

        if ('output' in activations) {
            const flat = activations.output.flat(Infinity);
            outputMin = Math.min(outputMin, ...flat);
            outputMax = Math.max(outputMax, ...flat);
        }

        const numHiddenLayers = state.weights.hidden_weights.length;
        const finalPreKey = `hidden_${numHiddenLayers}_pre`;
        if (finalPreKey in activations) {
            const flat = activations[finalPreKey].flat(Infinity);
            outputMin = Math.min(outputMin, ...flat);
            outputMax = Math.max(outputMax, ...flat);
        }

        return { min: outputMin, max: outputMax };
    },

    // Create layer row structure
    createLayerRow(layerName, forwardDims, backwardDims, hasGradients) {
        return `
            <div class="layer-row">
                <div class="layer-cell forward-cell" id="forward-${layerName.replace(/\s/g, '-')}">
                    <div class="layer-name">${layerName}</div>
                    <div class="layer-dims">${forwardDims}</div>
                    <div class="layer-type">Forward Pass</div>
                </div>
                <div class="layer-cell backward-cell ${hasGradients ? '' : 'disabled'}" id="backward-${layerName.replace(/\s/g, '-')}">
                    <div class="layer-name">${layerName}</div>
                    <div class="layer-dims">${backwardDims}</div>
                    <div class="layer-type">Backward Pass</div>
                </div>
            </div>
        `;
    },

    // Populate a layer cell with heatmaps
    populateLayerCell(layerName, weights, activations, gradients, minMax, outputMinMax, gradientMinMax,
                      inputKey, weightKey, biasKey, preActivKey, postActivKey) {
        const layerWeights = weights[weightKey];
        const layerBiases = weights[biasKey];
        this.populateLayerCellWithArrays(layerName, layerWeights, layerBiases,
            activations, gradients, minMax, outputMinMax, gradientMinMax, inputKey, weightKey, biasKey, preActivKey, postActivKey);
    },

    populateLayerCellWithArrays(layerName, layerWeights, layerBiases,
                                 activations, gradients, minMax, outputMinMax, gradientMinMax,
                                 inputKey, weightGradKey, biasGradKey, preActivKey, postActivKey) {
        const forwardId = `forward-${layerName.replace(/\s/g, '-')}`;
        const backwardId = `backward-${layerName.replace(/\s/g, '-')}`;

        // Forward Pass Cell
        const forwardContainer = document.getElementById(forwardId);
        if (forwardContainer && inputKey in activations) {
            let forwardHTML = `<div class="layer-name">${layerName}</div>`;

            // Top: Layer Input (averaged across batch)
            const inputActivation = this.averageAlongAxis0(activations[inputKey]);
            forwardHTML += this.renderHeatmap(inputActivation, forwardId, 'Input', minMax, true);

            // Middle: Weights and Biases
            forwardHTML += '<div class="weight-bias-container">';
            forwardHTML += '<div class="weight-matrix">';
            forwardHTML += this.renderHeatmap(layerWeights, forwardId, 'Weights', minMax, true);
            forwardHTML += '</div>';
            forwardHTML += '<div class="bias-matrix">';
            const biasAsRow = [layerBiases];
            forwardHTML += this.renderHeatmap(biasAsRow, forwardId, 'Bias', minMax, true);
            forwardHTML += '</div>';
            forwardHTML += '</div>';

            // Bottom: Pre-activation and Post-activation
            const activScale = outputMinMax || minMax;

            if (preActivKey in activations) {
                const preActiv = this.averageAlongAxis0(activations[preActivKey]);
                forwardHTML += this.renderHeatmap(preActiv, forwardId, 'Pre-Activation', activScale, true);
            }

            if (postActivKey in activations && postActivKey !== 'output') {
                const postActiv = this.averageAlongAxis0(activations[postActivKey]);
                forwardHTML += this.renderHeatmap(postActiv, forwardId, 'Post-Activation (GELU)', minMax, true);
            } else if (postActivKey === 'output' && postActivKey in activations) {
                const output = this.averageAlongAxis0(activations[postActivKey]);
                forwardHTML += this.renderHeatmap(output, forwardId, 'Output (Logits)', activScale, true);
            }

            forwardContainer.innerHTML = forwardHTML;
        }

        // Backward Pass Cell
        const backwardContainer = document.getElementById(backwardId);
        if (backwardContainer && weightGradKey in gradients) {
            let backwardHTML = `<div class="layer-name">${layerName}</div>`;

            backwardHTML += '<div class="matrix-label">Gradient Flow In</div>';

            // Weight gradients and Bias gradients
            backwardHTML += '<div class="weight-bias-container">';
            backwardHTML += '<div class="weight-matrix">';
            backwardHTML += this.renderHeatmap(gradients[weightGradKey], backwardId, 'Weight Gradients', gradientMinMax, false);
            backwardHTML += '</div>';
            backwardHTML += '<div class="bias-matrix">';
            const biasGradAsRow = [gradients[biasGradKey]];
            backwardHTML += this.renderHeatmap(biasGradAsRow, backwardId, 'Bias Gradients', gradientMinMax, false);
            backwardHTML += '</div>';
            backwardHTML += '</div>';

            backwardContainer.innerHTML = backwardHTML;
        }
    },

    // Display color scale legend
    displayColorScale(state) {
        const minMax = this.getGlobalMinMax(state);
        const absMax = Math.max(Math.abs(minMax.min), Math.abs(minMax.max));

        const outputMinMax = this.getOutputMinMax(state);
        const outputAbsMax = Math.max(Math.abs(outputMinMax.min), Math.abs(outputMinMax.max));

        const gradientMinMax = this.getGradientMinMax(state);
        const gradAbsMax = Math.max(Math.abs(gradientMinMax.min), Math.abs(gradientMinMax.max));

        const outputGradientMinMax = this.getOutputGradientMinMax(state);
        const outGradAbsMax = Math.max(Math.abs(outputGradientMinMax.min), Math.abs(outputGradientMinMax.max));

        const forwardGradient = 'linear-gradient(to right, ' +
            'rgb(255, 0, 0), rgb(255, 127, 127), rgb(255, 255, 255), rgb(127, 255, 127), rgb(0, 255, 0))';

        const backwardGradient = 'linear-gradient(to right, ' +
            'rgb(205, 140, 40), rgb(230, 197, 147), rgb(255, 255, 255), rgb(165, 190, 255), rgb(70, 130, 255))';

        const html = `
            <div class="color-scale-container" style="grid-template-columns: repeat(2, 1fr); gap: 15px;">
                <div class="scale-section">
                    <div class="scale-label">Forward Pass Scale (Red → White → Green)</div>
                    <div class="color-gradient" style="background: ${forwardGradient};"></div>
                    <div class="scale-markers">
                        <div class="scale-marker"><div class="marker-label">-Max</div><div class="marker-value">${(-absMax).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">-Half</div><div class="marker-value">${(-absMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">0</div><div class="marker-value">0.000</div></div>
                        <div class="scale-marker"><div class="marker-label">+Half</div><div class="marker-value">${(absMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">+Max</div><div class="marker-value">${absMax.toFixed(3)}</div></div>
                    </div>
                </div>
                <div class="scale-section">
                    <div class="scale-label">Gradients Scale (Orange → White → Blue)</div>
                    <div class="color-gradient" style="background: ${backwardGradient};"></div>
                    <div class="scale-markers">
                        <div class="scale-marker"><div class="marker-label">-Max</div><div class="marker-value">${(-gradAbsMax).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">-Half</div><div class="marker-value">${(-gradAbsMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">0</div><div class="marker-value">0.000</div></div>
                        <div class="scale-marker"><div class="marker-label">+Half</div><div class="marker-value">${(gradAbsMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">+Max</div><div class="marker-value">${gradAbsMax.toFixed(3)}</div></div>
                    </div>
                </div>
                <div class="scale-section">
                    <div class="scale-label">Output Logits Scale (Red → White → Green)</div>
                    <div class="color-gradient" style="background: ${forwardGradient};"></div>
                    <div class="scale-markers">
                        <div class="scale-marker"><div class="marker-label">-Max</div><div class="marker-value">${(-outputAbsMax).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">-Half</div><div class="marker-value">${(-outputAbsMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">0</div><div class="marker-value">0.000</div></div>
                        <div class="scale-marker"><div class="marker-label">+Half</div><div class="marker-value">${(outputAbsMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">+Max</div><div class="marker-value">${outputAbsMax.toFixed(3)}</div></div>
                    </div>
                </div>
                <div class="scale-section">
                    <div class="scale-label">Output Layer Gradients Scale (Orange → White → Blue)</div>
                    <div class="color-gradient" style="background: ${backwardGradient};"></div>
                    <div class="scale-markers">
                        <div class="scale-marker"><div class="marker-label">-Max</div><div class="marker-value">${(-outGradAbsMax).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">-Half</div><div class="marker-value">${(-outGradAbsMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">0</div><div class="marker-value">0.000</div></div>
                        <div class="scale-marker"><div class="marker-label">+Half</div><div class="marker-value">${(outGradAbsMax/2).toFixed(3)}</div></div>
                        <div class="scale-marker"><div class="marker-label">+Max</div><div class="marker-value">${outGradAbsMax.toFixed(3)}</div></div>
                    </div>
                </div>
            </div>
        `;

        document.getElementById('color-scale-legend').innerHTML = html;
    }
};

window.Heatmaps = Heatmaps;
