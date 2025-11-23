// Distribution analysis functions

const Distributions = {
    // Load and display distribution statistics
    async loadStats() {
        try {
            const data = await API.getDistributionStats();

            if (data.error) {
                console.error('Error loading distribution stats:', data.error);
                this.showError('Failed to load distribution statistics');
                return;
            }

            this.displayDistributions(data.weights, data.gradients);
            this.displayLayerStatsSummary(data);
            this.displayGradientNorms(data.gradient_norms);
        } catch (error) {
            console.error('Failed to load distribution stats:', error);
            this.showError(`Error: ${error.message}`);
        }
    },

    // Show error message
    showError(message) {
        const container = document.getElementById('distributions-container');
        if (container) {
            container.innerHTML = `<p style="color: #ef4444; text-align: center; padding: 40px;">${message}</p>`;
        }

        const summaryContainer = document.getElementById('layer-stats-summary');
        if (summaryContainer) {
            summaryContainer.innerHTML = `<p style="color: #ef4444;">${message}</p>`;
        }
    },

    // Display weight and gradient distributions in two-column layout
    displayDistributions(weightData, gradientData) {
        const container = document.getElementById('distributions-container');
        if (!container) return;

        if (!gradientData || Object.keys(gradientData).length === 0) {
            container.innerHTML = '<p style="color: #9ca3af; text-align: center; padding: 40px;">No gradient data available. Run a forward/backward pass first.</p>';
            return;
        }

        let html = '';

        // Create a row for each layer
        for (const [layerName, weightStats] of Object.entries(weightData)) {
            const gradStats = gradientData[layerName] || null;
            const layerId = layerName.replace(/\s/g, '-');
            const weightChartId = `weight-hist-${layerId}`;
            const gradChartId = `grad-hist-${layerId}`;

            html += `
                <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px;">
                    <!-- Weight Distribution (Left) -->
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #4ade80;">
                        <h4 style="color: #4ade80; margin-bottom: 10px;">${layerName} Weights</h4>
                        <div id="${weightChartId}" style="height: 200px;"></div>
                        <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; font-size: 0.85em;">
                            <div style="color: #9ca3af;">Mean: <span style="color: #60a5fa;">${weightStats.mean.toFixed(6)}</span></div>
                            <div style="color: #9ca3af;">Std: <span style="color: #f59e0b;">${weightStats.std.toFixed(6)}</span></div>
                            <div style="color: #9ca3af;">Range: <span style="color: #10b981;">[${weightStats.min.toFixed(4)}, ${weightStats.max.toFixed(4)}]</span></div>
                        </div>
                    </div>

                    <!-- Gradient Distribution (Right) -->
                    <div style="background: #1a1a1a; padding: 15px; border-radius: 8px; border-left: 4px solid #fb923c;">
            `;

            if (gradStats) {
                html += `
                        <h4 style="color: #fb923c; margin-bottom: 10px;">${layerName} Gradients</h4>
                        <div id="${gradChartId}" style="height: 200px;"></div>
                        <div style="display: grid; grid-template-columns: repeat(2, 1fr); gap: 10px; margin-top: 10px; font-size: 0.85em;">
                            <div style="color: #9ca3af;">Mean: <span style="color: #60a5fa;">${gradStats.mean.toFixed(8)}</span></div>
                            <div style="color: #9ca3af;">Std: <span style="color: #f59e0b;">${gradStats.std.toFixed(8)}</span></div>
                            <div style="color: #9ca3af;">L2 Norm: <span style="color: #10b981;">${gradStats.l2_norm.toFixed(6)}</span></div>
                            <div style="color: #9ca3af;">Range: <span style="color: #ec4899;">[${gradStats.min.toExponential(2)}, ${gradStats.max.toExponential(2)}]</span></div>
                        </div>
                `;
            } else {
                html += `
                        <h4 style="color: #fb923c; margin-bottom: 10px;">${layerName} Gradients</h4>
                        <p style="color: #9ca3af; padding: 60px 20px; text-align: center;">No gradient data</p>
                `;
            }

            html += `
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;

        // Render histograms after DOM update
        setTimeout(() => {
            for (const [layerName, weightStats] of Object.entries(weightData)) {
                const layerId = layerName.replace(/\s/g, '-');
                const weightChartId = `weight-hist-${layerId}`;
                const gradChartId = `grad-hist-${layerId}`;

                // Render weight histogram
                if (weightStats.values && weightStats.values.length > 0) {
                    Charts.renderHistogram(weightChartId, weightStats.values, '', 'rgb(74, 222, 128)');
                }

                // Render gradient histogram if available
                const gradStats = gradientData[layerName];
                if (gradStats && gradStats.values && gradStats.values.length > 0) {
                    Charts.renderHistogram(gradChartId, gradStats.values, '', 'rgb(251, 146, 60)');
                }
            }
        }, 0);
    },

    // Display layer statistics summary table
    displayLayerStatsSummary(data) {
        const container = document.getElementById('layer-stats-summary');
        if (!container) return;

        let html = `
            <table style="width: 100%; border-collapse: collapse; background: #1a1a1a; border-radius: 8px; overflow: hidden;">
                <thead>
                    <tr style="background: #2a2a2a;">
                        <th style="padding: 12px; text-align: left; color: #9ca3af; border-bottom: 1px solid #444;">Layer</th>
                        <th style="padding: 12px; text-align: center; color: #9ca3af; border-bottom: 1px solid #444;">Weight Mean</th>
                        <th style="padding: 12px; text-align: center; color: #9ca3af; border-bottom: 1px solid #444;">Weight Std</th>
                        <th style="padding: 12px; text-align: center; color: #9ca3af; border-bottom: 1px solid #444;">Grad Mean</th>
                        <th style="padding: 12px; text-align: center; color: #9ca3af; border-bottom: 1px solid #444;">Grad Std</th>
                        <th style="padding: 12px; text-align: center; color: #9ca3af; border-bottom: 1px solid #444;">Grad L2 Norm</th>
                    </tr>
                </thead>
                <tbody>
        `;

        for (const layerName of Object.keys(data.weights)) {
            const weightStats = data.weights[layerName];
            const gradStats = data.gradients[layerName] || {};

            html += `
                <tr style="border-bottom: 1px solid #333;">
                    <td style="padding: 10px; color: #e0e0e0; font-weight: 600;">${layerName}</td>
                    <td style="padding: 10px; text-align: center; color: #60a5fa;">${weightStats.mean.toFixed(6)}</td>
                    <td style="padding: 10px; text-align: center; color: #f59e0b;">${weightStats.std.toFixed(6)}</td>
                    <td style="padding: 10px; text-align: center; color: #60a5fa;">${gradStats.mean ? gradStats.mean.toExponential(3) : 'N/A'}</td>
                    <td style="padding: 10px; text-align: center; color: #f59e0b;">${gradStats.std ? gradStats.std.toExponential(3) : 'N/A'}</td>
                    <td style="padding: 10px; text-align: center; color: #10b981;">${gradStats.l2_norm ? gradStats.l2_norm.toFixed(6) : 'N/A'}</td>
                </tr>
            `;
        }

        html += '</tbody></table>';
        container.innerHTML = html;
    },

    // Display gradient norms over time
    displayGradientNorms(normHistory) {
        const container = document.getElementById('gradient-norms-chart');
        if (!container) return;

        if (!normHistory || Object.keys(normHistory).length === 0) {
            container.innerHTML = '<p style="color: #9ca3af;">No gradient norm history available yet. Train the model to see gradient flow over time.</p>';
            return;
        }

        Charts.renderGradientNorms('gradient-norms-chart', normHistory);
    }
};

window.Distributions = Distributions;
