// Main entry point and initialization

const App = {
    // Initialize the application
    async init() {
        console.log('Initializing Gradient Visualizer...');

        try {
            // Load all initial data in parallel
            await Promise.all([
                Display.loadModelState(),
                Display.loadDatasetInfo(),
                Display.loadTrainingStats(),
                Display.loadTrainerConfig(),
                Runs.loadRuns()
            ]);

            console.log('Gradient Visualizer initialized successfully');
        } catch (error) {
            console.error('Error initializing application:', error);
        }
    }
};

// Initialize on page load
window.addEventListener('load', () => App.init());

window.App = App;
