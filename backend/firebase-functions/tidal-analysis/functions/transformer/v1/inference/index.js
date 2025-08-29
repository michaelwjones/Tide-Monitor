const admin = require('firebase-admin');
const onnx = require('onnxruntime-node');
const fs = require('fs');
const path = require('path');

// Initialize Firebase Admin
if (!admin.apps.length) {
    admin.initializeApp();
}

// Transformer v1 Inferencer Class
class Transformerv1Inferencer {
    constructor() {
        this.session = null;
        this.normalizationParams = null;
        this.metadata = null;
        this.inputLength = 4320;  // 72 hours in minutes (seq2seq input)
        this.outputLength = 1440; // 24 hours in minutes (seq2seq output)
    }

    async initialize() {
        try {
            // Load ONNX model
            const modelPath = path.join(__dirname, 'transformer_tidal_v1.onnx');
            if (!fs.existsSync(modelPath)) {
                throw new Error(`ONNX model not found at ${modelPath}`);
            }
            
            this.session = await onnx.InferenceSession.create(modelPath);
            console.log('✅ Transformer ONNX model loaded successfully');

            // Load normalization parameters
            const normPath = path.join(__dirname, 'model_metadata.json');
            if (!fs.existsSync(normPath)) {
                throw new Error(`Metadata file not found at ${normPath}`);
            }
            
            const metadata = JSON.parse(fs.readFileSync(normPath, 'utf8'));
            this.normalizationParams = metadata.normalization;
            this.metadata = metadata;
            
            console.log('✅ Model metadata loaded:', {
                architecture: this.metadata.model_info.architecture,
                parameters: this.metadata.model_info.total_parameters,
                normalization: this.normalizationParams
            });

            return true;
        } catch (error) {
            console.error('❌ Failed to initialize Transformer model:', error);
            return false;
        }
    }

    normalize(values) {
        // Apply z-score normalization using training parameters
        const { mean, std } = this.normalizationParams;
        return values.map(val => (val - mean) / std);
    }

    denormalize(normalizedValues) {
        // Convert normalized predictions back to original scale
        const { mean, std } = this.normalizationParams;
        return normalizedValues.map(val => val * std + mean);
    }

    prepareInputSequence(waterLevels) {
        /**
         * Prepare input sequence for transformer inference
         * Requires exactly 4320 readings (72 hours)
         */
        let inputSequence = [...waterLevels];
        
        // Handle insufficient data by padding with mean value
        if (inputSequence.length < this.inputLength) {
            console.log(`⚠️  Input has ${inputSequence.length} readings, need ${this.inputLength}`);
            
            const meanValue = inputSequence.reduce((a, b) => a + b, 0) / inputSequence.length;
            const padLength = this.inputLength - inputSequence.length;
            
            console.log(`🔧 Padding with ${padLength} mean values (${meanValue.toFixed(1)})`);
            
            // Pad at the beginning with mean values
            const padding = new Array(padLength).fill(meanValue);
            inputSequence = [...padding, ...inputSequence];
        }
        
        // Handle excess data by taking most recent readings
        if (inputSequence.length > this.inputLength) {
            console.log(`📊 Truncating from ${inputSequence.length} to ${this.inputLength} readings`);
            inputSequence = inputSequence.slice(-this.inputLength);
        }

        console.log(`✅ Prepared input sequence: ${inputSequence.length} readings`);
        console.log(`   Range: ${Math.min(...inputSequence).toFixed(1)} - ${Math.max(...inputSequence).toFixed(1)} mm`);
        
        return inputSequence;
    }

    async predict24Hours(waterLevels) {
        if (!this.session) {
            throw new Error('Model not initialized');
        }

        try {
            console.log('🧠 Starting Transformer seq2seq prediction...');
            const startTime = Date.now();
            
            // Prepare input sequence (exactly 4320 readings)
            const inputSequence = this.prepareInputSequence(waterLevels);
            
            // Normalize input data
            const normalizedInput = this.normalize(inputSequence);
            
            // Create input tensor (shape: [1, 4320, 1])
            const inputTensor = new onnx.Tensor('float32', 
                Float32Array.from(normalizedInput), 
                [1, this.inputLength, 1]
            );

            console.log(`🔢 Input tensor shape: [${inputTensor.dims.join(', ')}]`);
            
            // Run seq2seq inference (single forward pass)
            const outputs = await this.session.run({ input: inputTensor });
            const inferenceTime = Date.now() - startTime;
            
            console.log(`⚡ Inference completed in ${inferenceTime}ms`);
            
            // Get output tensor (shape: [1, 1440, 1])
            const outputData = outputs.output.data;
            console.log(`📊 Output tensor shape: [${outputs.output.dims.join(', ')}]`);
            console.log(`📈 Raw output length: ${outputData.length}`);
            
            // Convert to array and denormalize
            const normalizedPredictions = Array.from(outputData);
            const predictions = this.denormalize(normalizedPredictions);
            
            console.log(`✅ Generated ${predictions.length} predictions`);
            console.log(`   Range: ${Math.min(...predictions).toFixed(1)} - ${Math.max(...predictions).toFixed(1)} mm`);
            
            // Create timestamped predictions
            const currentTime = new Date();
            const timestampedPredictions = predictions.map((prediction, index) => ({
                timestamp: new Date(currentTime.getTime() + (index + 1) * 60000).toISOString(),
                prediction: prediction,
                step: index + 1
            }));

            return {
                predictions: timestampedPredictions,
                metadata: {
                    inference_time_ms: inferenceTime,
                    input_length: inputSequence.length,
                    output_length: predictions.length,
                    model_architecture: 'seq2seq_transformer',
                    normalization: this.normalizationParams
                }
            };

        } catch (error) {
            console.error('❌ Prediction error:', error);
            throw error;
        }
    }
}

// Global instance
let inferencer = null;

// Initialize inferencer
async function getInferencer() {
    if (!inferencer) {
        inferencer = new Transformerv1Inferencer();
        const initialized = await inferencer.initialize();
        if (!initialized) {
            throw new Error('Failed to initialize Transformer inferencer');
        }
    }
    return inferencer;
}

// Firebase Cloud Function - Transformer v1 Prediction Runner
exports.runTransformerv1Prediction = admin.database().ref('/readings').onWrite(async (change, context) => {
    // Only run every 6 hours to generate 24-hour forecasts
    // This prevents excessive compute usage and provides regular updates
    const now = new Date();
    const hoursSinceStart = now.getHours();
    
    // Run at 00:00, 06:00, 12:00, 18:00
    if (hoursSinceStart % 6 !== 0 || now.getMinutes() > 30) {
        console.log(`⏰ Skipping prediction - scheduled for 00:00, 06:00, 12:00, 18:00 (current: ${now.getHours()}:${now.getMinutes().toString().padStart(2, '0')})`);
        return;
    }

    try {
        console.log('🚀 Starting Transformer v1 24-hour prediction run...');
        const startTime = Date.now();
        
        // Get inferencer instance
        const inferencer = await getInferencer();

        // Fetch last 72 hours of water level data from Firebase
        console.log('📡 Fetching historical data from Firebase...');
        const hoursAgo72 = Date.now() - (72 * 60 * 60 * 1000);

        const snapshot = await admin.database().ref('/readings')
            .orderByChild('published_at')
            .startAt(hoursAgo72)
            .once('value');

        const readings = snapshot.val();
        if (!readings) {
            console.log('❌ No readings found for prediction');
            return;
        }

        // Extract and sort water levels by timestamp
        const waterLevelData = [];
        Object.values(readings).forEach(reading => {
            if (reading.w && typeof reading.w === 'number' && reading.t) {
                waterLevelData.push({
                    timestamp: reading.t,
                    waterLevel: reading.w,
                    published_at: reading.published_at || Date.now()
                });
            }
        });

        // Sort by timestamp
        waterLevelData.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
        
        if (waterLevelData.length < 1440) { // Need at least 24 hours of data
            console.log(`⚠️  Insufficient data for prediction: ${waterLevelData.length} readings (need at least 1440)`);
            return;
        }

        console.log(`📊 Using ${waterLevelData.length} readings for prediction`);
        console.log(`📅 Time range: ${waterLevelData[0].timestamp} to ${waterLevelData[waterLevelData.length - 1].timestamp}`);

        // Extract water level values for prediction
        const waterLevelValues = waterLevelData.map(r => r.waterLevel);

        // Generate 24-hour forecast using seq2seq transformer
        const forecastResult = await inferencer.predict24Hours(waterLevelValues);

        // Prepare forecast data for storage
        const forecastData = {
            generated_at: admin.database.ServerValue.TIMESTAMP,
            model_version: 'transformer-v1',
            model_architecture: 'seq2seq_transformer',
            input_data_count: waterLevelValues.length,
            input_time_range: {
                start: waterLevelData[0].timestamp,
                end: waterLevelData[waterLevelData.length - 1].timestamp
            },
            forecast: forecastResult.predictions,
            forecast_count: forecastResult.predictions.length,
            metadata: forecastResult.metadata,
            generation_time_ms: Date.now() - startTime
        };

        // Store forecast with timestamp key for easy retrieval
        const timestampKey = new Date().toISOString().replace(/[:.]/g, '-');
        await admin.database().ref(`/tidal-analysis/transformer-v1-forecasts/${timestampKey}`)
            .set(forecastData);

        console.log(`✅ Transformer v1 forecast saved successfully`);
        console.log(`   Forecast key: ${timestampKey}`);
        console.log(`   Predictions: ${forecastResult.predictions.length}`);
        console.log(`   Total time: ${Date.now() - startTime}ms`);

        // Also store latest forecast at a fixed location for easy access
        await admin.database().ref('/tidal-analysis/latest-transformer-v1-forecast')
            .set(forecastData);

        console.log('📌 Latest forecast reference updated');

    } catch (error) {
        console.error('❌ Transformer v1 prediction error:', error);
        
        // Store error information for debugging
        const errorData = {
            generated_at: admin.database.ServerValue.TIMESTAMP,
            model_version: 'transformer-v1',
            error: error.message,
            stack: error.stack,
            timestamp: new Date().toISOString()
        };
        
        const timestampKey = new Date().toISOString().replace(/[:.]/g, '-');
        await admin.database().ref(`/tidal-analysis/transformer-v1-errors/${timestampKey}`)
            .set(errorData);
            
        console.log(`💾 Error information stored: ${timestampKey}`);
    }
});

// HTTP function for manual prediction testing
exports.testTransformerv1Prediction = async (req, res) => {
    try {
        console.log('🧪 Manual Transformer v1 prediction test');
        
        // CORS headers
        res.set('Access-Control-Allow-Origin', '*');
        if (req.method === 'OPTIONS') {
            res.set('Access-Control-Allow-Methods', 'POST');
            res.set('Access-Control-Allow-Headers', 'Content-Type');
            res.status(204).send('');
            return;
        }

        const inferencer = await getInferencer();
        
        // Use provided data or fetch from Firebase
        let waterLevelValues;
        if (req.body && req.body.waterLevels) {
            waterLevelValues = req.body.waterLevels;
            console.log(`📥 Using provided data: ${waterLevelValues.length} readings`);
        } else {
            // Fetch recent data from Firebase
            const hoursAgo72 = Date.now() - (72 * 60 * 60 * 1000);
            const snapshot = await admin.database().ref('/readings')
                .orderByChild('published_at')
                .startAt(hoursAgo72)
                .limitToLast(4320)
                .once('value');
            
            const readings = snapshot.val();
            if (!readings) {
                res.status(400).json({ error: 'No data available for prediction' });
                return;
            }
            
            const waterLevelData = [];
            Object.values(readings).forEach(reading => {
                if (reading.w && typeof reading.w === 'number') {
                    waterLevelData.push({
                        timestamp: reading.t,
                        waterLevel: reading.w
                    });
                }
            });
            
            waterLevelData.sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
            waterLevelValues = waterLevelData.map(r => r.waterLevel);
            console.log(`📡 Fetched ${waterLevelValues.length} readings from Firebase`);
        }

        // Generate prediction
        const forecastResult = await inferencer.predict24Hours(waterLevelValues);

        res.json({
            success: true,
            model_version: 'transformer-v1',
            input_count: waterLevelValues.length,
            output_count: forecastResult.predictions.length,
            forecast: forecastResult.predictions,
            metadata: forecastResult.metadata
        });

    } catch (error) {
        console.error('❌ Manual prediction error:', error);
        res.status(500).json({ 
            success: false,
            error: error.message 
        });
    }
};

// Export inferencer for testing
exports.getInferencer = getInferencer;