const admin = require('firebase-admin');
const onnx = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

// Initialize Firebase Admin
if (!admin.apps.length) {
    admin.initializeApp();
}

// LSTM v1 Inferencer Class
class LSTMv1Inferencer {
    constructor() {
        this.session = null;
        this.normalizationParams = null;
        this.metadata = null;
        this.sequenceLength = 4320; // 72 hours in minutes
    }

    async initialize() {
        try {
            // Load ONNX model
            const modelPath = path.join(__dirname, 'tidal_lstm.onnx');
            this.session = await onnx.InferenceSession.create(modelPath);
            console.log('ONNX model loaded successfully');

            // Load normalization parameters
            const normPath = path.join(__dirname, 'normalization_params.json');
            this.normalizationParams = JSON.parse(fs.readFileSync(normPath, 'utf8'));
            console.log('Normalization parameters loaded:', this.normalizationParams);

            // Load model metadata
            const metadataPath = path.join(__dirname, 'model_metadata.json');
            this.metadata = JSON.parse(fs.readFileSync(metadataPath, 'utf8'));
            console.log('Model metadata loaded:', this.metadata.model_type, this.metadata.version);

            return true;
        } catch (error) {
            console.error('Failed to initialize LSTM model:', error);
            return false;
        }
    }

    normalize(values) {
        // Apply z-score normalization using training parameters
        const { mean, std } = this.normalizationParams;
        return values.map(val => (val - mean) / std);
    }

    denormalize(normalizedValue) {
        // Convert normalized prediction back to original scale
        const { mean, std } = this.normalizationParams;
        return normalizedValue * std + mean;
    }

    async predict(sequence) {
        if (!this.session) {
            throw new Error('Model not initialized');
        }

        try {
            // Ensure sequence is the right length
            let inputSequence = [...sequence];
            
            // Pad or truncate to exact sequence length
            if (inputSequence.length < this.sequenceLength) {
                // Pad with -1 values at the beginning
                const padding = new Array(this.sequenceLength - inputSequence.length).fill(-1);
                inputSequence = [...padding, ...inputSequence];
            } else if (inputSequence.length > this.sequenceLength) {
                // Take the last sequenceLength values
                inputSequence = inputSequence.slice(-this.sequenceLength);
            }

            // Normalize input data
            const normalizedInput = this.normalize(inputSequence);

            // Create tensor (shape: [1, sequence_length, 1])
            const inputTensor = new onnx.Tensor('float32', 
                Float32Array.from(normalizedInput), 
                [1, this.sequenceLength, 1]
            );

            // Run inference
            const outputs = await this.session.run({ input: inputTensor });
            
            // Get prediction and denormalize
            const normalizedPrediction = outputs.output.data[0];
            const prediction = this.denormalize(normalizedPrediction);

            return prediction;

        } catch (error) {
            console.error('Prediction error:', error);
            throw error;
        }
    }

    async predictIterative24Hours(initialWaterLevels) {
        console.log('Starting 24-hour iterative prediction...');
        
        // Prepare initial sequence (last 72 hours)
        let currentSequence = [...initialWaterLevels];
        
        // Ensure we have at most sequenceLength values
        if (currentSequence.length > this.sequenceLength) {
            currentSequence = currentSequence.slice(-this.sequenceLength);
            console.log(`Truncated input to last ${this.sequenceLength} values`);
        }

        console.log(`Starting with ${currentSequence.length} historical values`);

        const predictions = [];
        const startTime = new Date();

        // Generate 1440 predictions (24 hours)
        for (let i = 0; i < 1440; i++) {
            try {
                // Predict next minute
                const nextPrediction = await this.predict(currentSequence);
                
                // Calculate prediction timestamp (1 minute from last data point)
                const predictionTime = new Date(startTime.getTime() + (i + 1) * 60000);
                
                predictions.push({
                    timestamp: predictionTime.toISOString(),
                    prediction: nextPrediction,
                    step: i + 1
                });

                // Update sequence: add prediction and maintain length
                currentSequence.push(nextPrediction);
                if (currentSequence.length > this.sequenceLength) {
                    currentSequence = currentSequence.slice(-this.sequenceLength);
                }

                // Log progress every 240 steps (4 hours)
                if ((i + 1) % 240 === 0) {
                    console.log(`Completed ${i + 1}/1440 predictions (${Math.round((i + 1) / 1440 * 100)}%)`);
                }

            } catch (error) {
                console.error(`Error at prediction step ${i + 1}:`, error);
                // Store error information instead of crashing
                predictions.push({
                    timestamp: new Date(startTime.getTime() + (i + 1) * 60000).toISOString(),
                    prediction: null,
                    error: error.message,
                    step: i + 1
                });
            }
        }

        console.log(`Completed 24-hour prediction: ${predictions.length} steps`);
        return predictions;
    }
}

// Global instance
let inferencer = null;

// Initialize inferencer
async function getInferencer() {
    if (!inferencer) {
        inferencer = new LSTMv1Inferencer();
        const initialized = await inferencer.initialize();
        if (!initialized) {
            throw new Error('Failed to initialize LSTM inferencer');
        }
    }
    return inferencer;
}

// Firebase Cloud Function - LSTM v1 Prediction Runner
exports.runLSTMv1Prediction = admin.database().ref('/readings').onWrite(async (change, context) => {
    // Only run every 6 hours to generate 24-hour forecasts
    const now = new Date();
    const lastRunTime = now.getHours() % 6;
    
    if (lastRunTime !== 0) {
        console.log('Skipping LSTM prediction - not scheduled time');
        return;
    }

    try {
        console.log('Starting LSTM v1 24-hour prediction run...');
        
        // Get inferencer instance
        const inferencer = await getInferencer();

        // Fetch last 72 hours of water level data from Firebase
        const endTime = admin.database.ServerValue.TIMESTAMP;
        const startTime = Date.now() - (72 * 60 * 60 * 1000); // 72 hours ago

        const snapshot = await admin.database().ref('/readings')
            .orderByChild('timestamp')
            .startAt(startTime)
            .endAt(endTime)
            .once('value');

        const readings = snapshot.val();
        if (!readings) {
            console.log('No readings found for prediction');
            return;
        }

        // Extract water levels and sort by timestamp
        const waterLevels = [];
        Object.values(readings).forEach(reading => {
            if (reading.w && typeof reading.w === 'number') {
                waterLevels.push({
                    timestamp: reading.t,
                    waterLevel: reading.w
                });
            }
        });

        waterLevels.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        
        if (waterLevels.length < 1440) { // Need at least 24 hours of data
            console.log(`Insufficient data for prediction: ${waterLevels.length} readings`);
            return;
        }

        console.log(`Using ${waterLevels.length} readings for prediction`);

        // Extract just the water level values
        const waterLevelValues = waterLevels.map(r => r.waterLevel);

        // Generate 24-hour forecast
        const forecast = await inferencer.predictIterative24Hours(waterLevelValues);

        // Store forecast in Firebase
        const forecastData = {
            generated_at: admin.database.ServerValue.TIMESTAMP,
            model_version: 'lstm-v1',
            input_data_count: waterLevelValues.length,
            input_time_range: {
                start: waterLevels[0].timestamp,
                end: waterLevels[waterLevels.length - 1].timestamp
            },
            forecast: forecast,
            forecast_count: forecast.length
        };

        // Store with timestamp key for easy retrieval
        const timestampKey = new Date().toISOString().replace(/[:.]/g, '-');
        await admin.database().ref(`/tidal-analysis/lstm-v1-forecasts/${timestampKey}`)
            .set(forecastData);

        console.log(`LSTM v1 forecast saved: ${forecast.length} predictions`);
        console.log(`Forecast key: ${timestampKey}`);

    } catch (error) {
        console.error('LSTM v1 prediction error:', error);
        
        // Store error information
        const errorData = {
            generated_at: admin.database.ServerValue.TIMESTAMP,
            model_version: 'lstm-v1',
            error: error.message,
            stack: error.stack
        };
        
        const timestampKey = new Date().toISOString().replace(/[:.]/g, '-');
        await admin.database().ref(`/tidal-analysis/lstm-v1-errors/${timestampKey}`)
            .set(errorData);
    }
});

// Export inferencer for testing
exports.getInferencer = getInferencer;