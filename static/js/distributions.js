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

            this.displayWeightDistributions(data.weights);
            this.displayGradientDistributions(data.gradients);
            this.displayLayerStatsSummary(data);
            this.displayGradientNorms(data.gradient_norms);
        } catch (error) {
            console.error('Failed to load distribution stats:', error);
            this.showError(`Error: ${error.message}`);
        }
    },

    // Show error message
    showError(message) {
        const containers = [
            'weight-distributions',
            'gradient-distributions',
            'layer-stats-summary'
        ];

        containers.forEach(id => {
            const container = document.getElementById(id);
            if (container) {
                container.innerHTML = `<p style="color: #ef4444;">${message}</p>`;
            }
        });
    },

    // Display weight distributions
    displayWeightDistributions(weightData) {
        const container = document.getElementById('weight-distributions');
        if (!container) return;

        let html = '';

        for (const [layerName, stats] of Object.entries(weightData)) {
            const chartId = `weight-hist-${layerName.replace(/\s/g, '-')}`;
            html += `
                <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #e0e0e0; margin-bottom: 10px;">${layerName}</h4>
                    <div id="${chartId}" style="height: 200px;"></div>
                    <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; margin-top: 10px; font-size: 0.85em;">
                        <div style="color: #9ca3af;">Mean: <span style="color: #60a5fa;">${stats.mean.toFixed(6)}</span></div>
                        <div style="color: #9ca3af;">Std: <span style="color: #f59e0b;">${stats.std.toFixed(6)}</span></div>
                        <div style="color: #9ca3af;">Range: <span style="color: #10b981;">[${stats.min.toFixed(4)}, ${stats.max.toFixed(4)}]</span></div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;

        // Render histograms after DOM update
        setTimeout(() => {
            for (const [layerName, stats] of Object.entries(weightData)) {
                const chartId = `weight-hist-${layerName.replace(/\s/g, '-')}`;
                if (stats.values && stats.values.length > 0) {
                    Charts.renderHistogram(chartId, stats.values, '', 'rgb(59, 130, 246)');
                }
            }
        }, 0);
    },

    // Display gradient distributions
    displayGradientDistributions(gradientData) {
        const container = document.getElementById('gradient-distributions');
        if (!container) return;

        if (!gradientData || Object.keys(gradientData).length === 0) {
            container.innerHTML = '<p style="color: #9ca3af;">No gradient data available. Run a forward/backward pass first.</p>';
            return;
        }

        let html = '';

        for (const [layerName, stats] of Object.entries(gradientData)) {
            const chartId = `grad-hist-${layerName.replace(/\s/g, '-')}`;
            html += `
                <div style="background: #1a1a1a; padding: 15px; border-radius: 8px;">
                    <h4 style="color: #e0e0e0; margin-bottom: 10px;">${layerName}</h4>
                    <div id="${chartId}" style="height: 200px;"></div>
                    <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 10px; margin-top: 10px; font-size: 0.85em;">
                        <div style="color: #9ca3af;">Mean: <span style="color: #60a5fa;">${stats.mean.toFixed(8)}</span></div>
                        <div style="color: #9ca3af;">Std: <span style="color: #f59e0b;">${stats.std.toFixed(8)}</span></div>
                        <div style="color: #9ca3af;">L2 Norm: <span style="color: #10b981;">${stats.l2_norm.toFixed(6)}</span></div>
                        <div style="color: #9ca3af;">Range: <span style="color: #ec4899;">[${stats.min.toExponential(2)}, ${stats.max.toExponential(2)}]</span></div>
                    </div>
                </div>
            `;
        }

        container.innerHTML = html;

        // Render histograms after DOM update
        setTimeout(() => {
            for (const [layerName, stats] of Object.entries(gradientData)) {
                const chartId = `grad-hist-${layerName.replace(/\s/g, '-')}`;
                if (stats.values && stats.values.length > 0) {
                    Charts.renderHistogram(chartId, stats.values, '', 'rgb(245, 158, 11)');
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
