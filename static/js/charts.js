// Chart rendering functions using Plotly

const Charts = {
    // Color palette for multiple runs
    colors: [
        'rgb(138, 43, 226)',  // Purple
        'rgb(59, 130, 246)',  // Blue
        'rgb(16, 185, 129)',  // Green
        'rgb(245, 158, 11)',  // Orange
        'rgb(239, 68, 68)',   // Red
        'rgb(236, 72, 153)',  // Pink
        'rgb(139, 92, 246)',  // Indigo
        'rgb(6, 182, 212)',   // Cyan
    ],

    // Common layout settings
    getBaseLayout(title) {
        return {
            title: {
                text: title,
                font: { size: 16, color: '#e0e0e0' }
            },
            plot_bgcolor: '#2a2a2a',
            paper_bgcolor: '#1a1a1a',
            font: { color: '#e0e0e0' },
            margin: { l: 60, r: 30, t: 50, b: 50 },
            hovermode: 'closest'
        };
    },

    // Update multi-run comparison chart
    updateMultiRunChart(runLosses, currentStats = null) {
        const traces = [];
        let colorIndex = 0;

        console.log('Run losses data:', runLosses);
        console.log('Current stats:', currentStats);

        for (const [runId, runData] of Object.entries(runLosses)) {
            if (!runData || !runData.losses || runData.losses.length === 0) continue;

            const steps = Array.from({length: runData.losses.length}, (_, i) => i + 1);
            const color = this.colors[colorIndex % this.colors.length];

            traces.push({
                x: steps,
                y: runData.losses,
                type: 'scatter',
                mode: 'lines+markers',
                name: `${runData.name} (LR: ${runData.config.learning_rate})`,
                line: { color: color, width: 2 },
                marker: { color: color, size: 4 }
            });

            colorIndex++;
        }

        // Fallback to single run chart if no runs
        if (traces.length === 0 && currentStats && currentStats.losses && currentStats.losses.length > 0) {
            console.log('Falling back to session data');
            this.updateSingleRunChart(currentStats);
            return;
        }

        if (traces.length === 0) {
            document.getElementById('loss-chart').innerHTML =
                '<div style="text-align: center; color: #9ca3af; padding: 50px;">No training data yet. Create a run and start training!</div>';
            return;
        }

        const layout = {
            ...this.getBaseLayout(traces.length > 1 ? 'Loss Comparison Across Training Runs' : 'Loss Over Training Steps'),
            xaxis: { title: 'Training Step', gridcolor: '#444', color: '#e0e0e0' },
            yaxis: { title: 'Loss', gridcolor: '#444', color: '#e0e0e0' },
            showlegend: traces.length > 1,
            legend: { bgcolor: 'rgba(0,0,0,0.5)', font: { color: '#e0e0e0' } }
        };

        const config = { responsive: true, displayModeBar: true, displaylogo: false };

        Plotly.react('loss-chart', traces, layout, config);
    },

    // Update single run chart
    updateSingleRunChart(stats) {
        if (stats.losses.length === 0) {
            document.getElementById('loss-chart').innerHTML =
                '<div style="text-align: center; color: #9ca3af; padding: 50px;">No training data yet. Start training to see loss curve.</div>';
            return;
        }

        const steps = Array.from({length: stats.losses.length}, (_, i) => i + 1);
        const losses = stats.losses;

        const trace = {
            x: steps,
            y: losses,
            type: 'scatter',
            mode: 'lines+markers',
            name: 'Current Session',
            line: { color: 'rgb(138, 43, 226)', width: 2 },
            marker: { color: 'rgb(138, 43, 226)', size: 4 }
        };

        const layout = {
            ...this.getBaseLayout('Loss Over Training Steps'),
            xaxis: { title: 'Training Step', gridcolor: '#444', color: '#e0e0e0' },
            yaxis: { title: 'Loss', gridcolor: '#444', color: '#e0e0e0' }
        };

        const config = { responsive: true, displayModeBar: true, displaylogo: false };

        Plotly.react('loss-chart', [trace], layout, config);
    },

    // Render distribution histogram
    renderHistogram(containerId, data, title, color = 'rgb(138, 43, 226)') {
        const trace = {
            x: data,
            type: 'histogram',
            marker: { color: color },
            opacity: 0.75
        };

        const layout = {
            ...this.getBaseLayout(title),
            xaxis: { title: 'Value', gridcolor: '#444', color: '#e0e0e0' },
            yaxis: { title: 'Frequency', gridcolor: '#444', color: '#e0e0e0' },
            bargap: 0.05
        };

        const config = { responsive: true, displayModeBar: false };

        Plotly.react(containerId, [trace], layout, config);
    },

    // Render gradient norms over time
    renderGradientNorms(containerId, normHistory) {
        const traces = [];
        let colorIndex = 0;

        for (const [layerName, norms] of Object.entries(normHistory)) {
            const steps = Array.from({length: norms.length}, (_, i) => i + 1);
            const color = this.colors[colorIndex % this.colors.length];

            traces.push({
                x: steps,
                y: norms,
                type: 'scatter',
                mode: 'lines',
                name: layerName,
                line: { color: color, width: 2 }
            });

            colorIndex++;
        }

        if (traces.length === 0) {
            document.getElementById(containerId).innerHTML =
                '<div style="text-align: center; color: #9ca3af; padding: 30px;">No gradient norm data yet.</div>';
            return;
        }

        const layout = {
            ...this.getBaseLayout('Gradient Norms Over Time'),
            xaxis: { title: 'Training Step', gridcolor: '#444', color: '#e0e0e0' },
            yaxis: { title: 'L2 Norm', gridcolor: '#444', color: '#e0e0e0', type: 'log' },
            showlegend: true,
            legend: { bgcolor: 'rgba(0,0,0,0.5)', font: { color: '#e0e0e0' } }
        };

        const config = { responsive: true, displayModeBar: true, displaylogo: false };

        Plotly.react(containerId, traces, layout, config);
    },

    // Render layer-wise statistics comparison
    renderLayerStats(containerId, layerStats) {
        const layers = Object.keys(layerStats);
        const means = layers.map(l => layerStats[l].mean);
        const stds = layers.map(l => layerStats[l].std);

        const trace1 = {
            x: layers,
            y: means,
            name: 'Mean',
            type: 'bar',
            marker: { color: 'rgb(59, 130, 246)' }
        };

        const trace2 = {
            x: layers,
            y: stds,
            name: 'Std Dev',
            type: 'bar',
            marker: { color: 'rgb(245, 158, 11)' }
        };

        const layout = {
            ...this.getBaseLayout('Layer-wise Statistics'),
            xaxis: { title: 'Layer', gridcolor: '#444', color: '#e0e0e0' },
            yaxis: { title: 'Value', gridcolor: '#444', color: '#e0e0e0' },
            barmode: 'group',
            showlegend: true,
            legend: { bgcolor: 'rgba(0,0,0,0.5)', font: { color: '#e0e0e0' } }
        };

        const config = { responsive: true, displayModeBar: false };

        Plotly.react(containerId, [trace1, trace2], layout, config);
    }
};

window.Charts = Charts;
