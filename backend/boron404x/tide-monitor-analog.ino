// Tide Monitor using Particle Boron 404X and HRXL-MaxSonar MB7360
// This code reads analog voltage from the ultrasonic sensor every minute
// and publishes data immediately to the Particle cloud (then to Firebase)
// Includes offline RAM storage for when cellular connectivity is lost

// Pin configuration
const int SENSOR_PIN = A5;  // Analog pin connected to the rangefinder's voltage output

// Installation configuration
const int MOUNT_HEIGHT = int(8 * 12 * 25.4);  // 8' in millimeters (* 25.4)

// System variables
const int NUM_SAMPLES = 512;                 // Reduced from 1024 to prevent crashes
const int READING_INTERVAL_MS = 60000;       // Take reading every 60 seconds (1 minute)
const int SAMPLE_DELAY_MS = 50;              // Delay between samples

// Publishing constraints
const int MAX_PUBLISH_SIZE = 1024;           // Maximum bytes per Particle publish

// Offline storage configuration
const int MAX_STORED_READINGS = 400;         // Maximum readings to store offline (~6.7 hours)
const int MAX_JSON_LENGTH = 120;             // Maximum length of JSON string per reading

// Variables for data storage and publication
int depthInMm = 0;
int waterLevelAverage = 0;
int waveHeightPercentile = 0;
int waveHeightEnvelope = 0;
int waveHeightBinning = 0;
int waterLevelPercentile = 0;
int waterLevelEnvelope = 0;
int waterLevelBinning = 0;

// Variables for timing
unsigned long lastReadingTime = 0;

// Reading counter
int consecutiveReadings = 0;

// Offline storage variables (in RAM) - simple flat array of JSON strings
char offlineReadings[MAX_STORED_READINGS][MAX_JSON_LENGTH];
int storedReadingCount = 0;      // Number of readings currently stored

// Forward declarations
void takeMeasurement();
void publishAllReadings();
void storeReading(String jsonString);
int calculateAverage(int samples[], int numSamples);
int calculateWaveHeightPercentile(int samples[], int numSamples, int* waterLevel);
int calculateWaveHeightEnvelope(int samples[], int numSamples, int* waterLevel);
int calculateWaveHeightBinning(int samples[], int numSamples, int* waterLevel);
int adcToDepthMm(int adcReading);

void setup() {
    // Initialize serial communication for debugging
    Serial.begin(9600);
    Serial.println("Tide monitor initializing...");
    
    // Initialize offline storage
    storedReadingCount = 0;
    
    // Connect to Particle cloud
    Particle.connect();
    waitFor(Particle.connected, 30000);
    
    if (Particle.connected()) {
        Particle.publish("tideMonitor/status", "Device initialized", PRIVATE);
        Serial.println("Connected to Particle cloud!");
    }
    
    Serial.printlnf("Offline storage capacity: %d readings (~%.1f hours)", 
                   MAX_STORED_READINGS, (float)MAX_STORED_READINGS / 60.0);
    
    // Initialize timing to trigger immediate first reading
    lastReadingTime = 0;
}

void loop() {
    // Check if it's time to take a reading
    if (millis() - lastReadingTime >= READING_INTERVAL_MS) {
        // Update timestamp first to ensure consistent intervals
        lastReadingTime = millis();
        
        takeMeasurement();
        publishAllReadings();
    }
    
    // Small delay to prevent busy-waiting
    delay(100);
}

void takeMeasurement() {
    Serial.println("Taking measurement...");
    
    // Take multiple samples for noise reduction
    int samples[NUM_SAMPLES];
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        samples[i] = analogRead(SENSOR_PIN);
        delay(SAMPLE_DELAY_MS);  // Configurable delay between samples
    }
    
    // Filter out invalid sensor readings and convert to distance (mm)
    int validDistances[NUM_SAMPLES];  // Now stores distance values in mm
    int validSampleCount = 0;
    
    for (int i = 0; i < NUM_SAMPLES; i++) {
        int distanceMm = adcToDepthMm(samples[i]);
        if (distanceMm >= 300 && distanceMm <= 4000) {
            validDistances[validSampleCount] = distanceMm;  // Store converted distance
            validSampleCount++;
        }
    }
    
    Serial.printlnf("Filtered samples: %d valid out of %d total", validSampleCount, NUM_SAMPLES);
    
    // Calculate simple average of valid distance samples
    int averageDistance = calculateAverage(validDistances, validSampleCount);
    
    // Calculate wave height using 10-90 percentile analysis
    waveHeightPercentile = calculateWaveHeightPercentile(validDistances, validSampleCount, &waterLevelPercentile);
    
    // Calculate wave height using envelope analysis
    waveHeightEnvelope = calculateWaveHeightEnvelope(validDistances, validSampleCount, &waterLevelEnvelope);
    
    // Calculate wave height using binning analysis
    waveHeightBinning = calculateWaveHeightBinning(validDistances, validSampleCount, &waterLevelBinning);
    
    // Use the average distance directly (already converted)
    depthInMm = averageDistance;
    
    // Calculate water level based on your installation
    waterLevelAverage = MOUNT_HEIGHT - depthInMm;
    
    // Increment reading counter
    consecutiveReadings++;
    
    // Create JSON string for this reading
    String timestamp = Time.format(TIME_FORMAT_ISO8601_FULL);
    String jsonData = "{\"t\":\"" + timestamp + "\",";
    jsonData += "\"w\":" + String(waterLevelAverage) + ",";
    jsonData += "\"hp\":" + String(waveHeightPercentile) + ",";
    jsonData += "\"he\":" + String(waveHeightEnvelope) + ",";
    jsonData += "\"hb\":" + String(waveHeightBinning) + ",";
    jsonData += "\"wp\":" + String(waterLevelPercentile) + ",";
    jsonData += "\"we\":" + String(waterLevelEnvelope) + ",";
    jsonData += "\"wb\":" + String(waterLevelBinning) + ",";
    jsonData += "\"vs\":" + String(validSampleCount) + "}";  // Include valid sample count
    
    // Store this reading
    storeReading(jsonData);
    
    // Debug information
    Serial.printlnf("Reading #%d: Depth: %dmm, Water Level: %dmm, Wave Height (P/E/B): %d/%d/%dmm, Method Levels (P/E/B): %d/%d/%dmm (Valid: %d/%d, Stored: %d)", 
                   consecutiveReadings, depthInMm, waterLevelAverage, waveHeightPercentile, waveHeightEnvelope, waveHeightBinning, 
                   waterLevelPercentile, waterLevelEnvelope, waterLevelBinning, validSampleCount, NUM_SAMPLES, storedReadingCount);
}

void publishAllReadings() {
    if (storedReadingCount == 0) {
        Serial.println("No readings to publish");
        return;
    }
    
    Serial.printlnf("Publishing all %d stored readings in batches...", storedReadingCount);
    
    // Connect to Particle cloud if not already connected
    if (!Particle.connected()) {
        Serial.println("Connecting to Particle cloud...");
        Particle.connect();
        
        // Wait for connection with timeout
        unsigned long connectStartTime = millis();
        while (!Particle.connected() && (millis() - connectStartTime < 20000)) {
            delay(100);
        }
    }
    
    if (!Particle.connected()) {
        Serial.printlnf("No connectivity - %d readings remain stored", storedReadingCount);
        return;
    }
    
    // Publish readings in batches that fit within the size limit
    while (storedReadingCount > 0) {
        // Build a batch that fits within MAX_PUBLISH_SIZE
        String batch;
        int readingsInBatch = 0;
        
        if (storedReadingCount == 1) {
            // Single reading - send as individual JSON object
            batch = String(offlineReadings[0]);
            readingsInBatch = 1;
        } else {
            // Multiple readings - send as JSON array
            batch = "{\"readings\":[";
            
            for (int i = 0; i < storedReadingCount; i++) {
                String currentReading = String(offlineReadings[i]);
                
                // Check if adding this reading would exceed the limit
                String testBatch = batch;
                if (readingsInBatch > 0) {
                    testBatch += ",";
                }
                testBatch += currentReading;
                
                if (readingsInBatch == 0) {
                    // First reading in batch - always include it
                    batch += currentReading;
                    readingsInBatch = 1;
                } else if (testBatch.length() + 2 <= MAX_PUBLISH_SIZE) { // +2 for closing ]}
                    // Reading fits, add it to batch
                    batch += "," + currentReading;
                    readingsInBatch++;
                } else {
                    // Reading doesn't fit, stop building this batch
                    break;
                }
            }
            
            batch += "]}";
        }
        
        // Try to publish the batch
        bool success = Particle.publish("tideMonitor/reading", batch, PRIVATE);
        
        if (success) {
            Serial.printlnf("Published batch of %d readings (%d bytes)", readingsInBatch, batch.length());
            
            // Remove the published readings from the front of the array
            for (int i = 0; i < storedReadingCount - readingsInBatch; i++) {
                strcpy(offlineReadings[i], offlineReadings[i + readingsInBatch]);
            }
            storedReadingCount -= readingsInBatch;
            
            Serial.printlnf("%d readings remaining in storage", storedReadingCount);
            
            // If all readings are now published, break out of loop
            if (storedReadingCount == 0) {
                break;
            }
            
            if (storedReadingCount > 0) {
                delay(1000);  // Rate limit: max 1 publish per second
            }
        } else {
            Serial.printlnf("Failed to publish batch - stopping (keeping %d readings in storage)", storedReadingCount);
            break;
        }
        
        // Check if we're still connected
        if (!Particle.connected()) {
            Serial.println("Lost connection during batch publish");
            break;
        }
    }
    
    if (storedReadingCount == 0) {
        Serial.println("All readings published successfully");
    }
}

int calculateAverage(int samples[], int numSamples) {
    // Calculate simple average of all distance samples (already in mm)
    long sum = 0;
    for (int i = 0; i < numSamples; i++) {
        sum += samples[i];
    }
    
    float average = (float)sum / numSamples;
    return (int)round(average);
}

int calculateWaveHeightPercentile(int samples[], int numSamples, int* waterLevel) {
    // Wave height calculation using 10-90 percentile method
    // This method is robust against outliers and provides a statistical measure of wave amplitude
    // Input: samples[] contains distance values in mm (already converted)
    
    // Create a copy of samples for sorting (don't modify original) - use heap allocation
    int* sortedSamples = (int*)malloc(numSamples * sizeof(int));
    if (sortedSamples == NULL) {
        Serial.println("Error: Failed to allocate memory for percentile calculation");
        *waterLevel = 0;
        return 0;
    }
    
    for (int i = 0; i < numSamples; i++) {
        sortedSamples[i] = samples[i];
    }
    
    // Shell sort - much more efficient than insertion sort for large arrays
    for (int gap = numSamples / 2; gap > 0; gap /= 2) {
        for (int i = gap; i < numSamples; i++) {
            int temp = sortedSamples[i];
            int j;
            for (j = i; j >= gap && sortedSamples[j - gap] > temp; j -= gap) {
                sortedSamples[j] = sortedSamples[j - gap];
            }
            sortedSamples[j] = temp;
        }
    }
    
    // Calculate 10th and 90th percentile indices
    int p10_index = (int)(0.1 * numSamples);
    int p90_index = (int)(0.9 * numSamples);
    
    // Ensure indices are within bounds
    if (p10_index < 0) p10_index = 0;
    if (p90_index >= numSamples) p90_index = numSamples - 1;
    
    // Get the 10th and 90th percentile distance values (already in mm)
    int p10_distance = sortedSamples[p10_index];  // Shorter distances = higher water (crests)
    int p90_distance = sortedSamples[p90_index];  // Longer distances = lower water (troughs)
    
    // Calculate trough water level (longer distance = lower water level)
    *waterLevel = MOUNT_HEIGHT - p90_distance;
    
    // Wave height is the difference between trough and crest distances
    int height = p90_distance - p10_distance;
    
    // Clean up allocated memory
    free(sortedSamples);
    
    // Return calculated height (should be positive if logic is correct)
    return height;
}

void storeReading(String jsonString) {
    if (storedReadingCount >= MAX_STORED_READINGS) {
        Serial.println("Warning: Storage full, discarding oldest readings");
        
        // Shift all readings down by one to make room for new reading
        for (int i = 0; i < MAX_STORED_READINGS - 1; i++) {
            strcpy(offlineReadings[i], offlineReadings[i + 1]);
        }
        storedReadingCount = MAX_STORED_READINGS - 1;
    }
    
    // Store the new reading at the end
    jsonString.toCharArray(offlineReadings[storedReadingCount], MAX_JSON_LENGTH);
    storedReadingCount++;
}

int adcToDepthMm(int adcReading) {
    // Convert ADC reading to depth in millimeters
    // Formula: divide by 4, multiply by 5, round to nearest integer
    return (int)round((adcReading / 4.0) * 5.0);
}

int calculateWaveHeightEnvelope(int samples[], int numSamples, int* waterLevel) {
    // Wave height calculation using envelope analysis
    // This method finds local peaks and troughs to determine the signal envelope
    // Input: samples[] contains distance values in mm (already converted)
    
    // Arrays to store detected peaks and troughs - use heap allocation
    int* peaks = (int*)malloc(numSamples * sizeof(int));
    int* troughs = (int*)malloc(numSamples * sizeof(int));
    if (peaks == NULL || troughs == NULL) {
        Serial.println("Error: Failed to allocate memory for envelope calculation");
        if (peaks) free(peaks);
        if (troughs) free(troughs);
        *waterLevel = 0;
        return 0;
    }
    
    int peakCount = 0;
    int troughCount = 0;
    
    // Minimum separation between peaks/troughs (prevents noise from creating false peaks)
    int minSeparation = 5;
    
    // Find local maxima (peaks) and minima (troughs)
    for (int i = minSeparation; i < numSamples - minSeparation; i++) {
        bool isPeak = true;
        bool isTrough = true;
        
        // Check if this point is higher/lower than surrounding points
        for (int j = i - minSeparation; j <= i + minSeparation; j++) {
            if (j != i) {
                if (samples[j] >= samples[i]) {
                    isPeak = false;
                }
                if (samples[j] <= samples[i]) {
                    isTrough = false;
                }
            }
        }
        
        // Store peaks and troughs
        if (isPeak) {
            peaks[peakCount] = samples[i];  // Higher distances = lower water (troughs)
            peakCount++;
        }
        if (isTrough) {
            troughs[troughCount] = samples[i];  // Lower distances = higher water (crests)
            troughCount++;
        }
    }
    
    // Return 0 if envelope analysis failed to find peaks or troughs
    // This indicates the method failed rather than masking with a fallback
    if (peakCount == 0 || troughCount == 0) {
        free(peaks);
        free(troughs);
        *waterLevel = 0;
        return 0;
    }
    
    // Calculate average of upper envelope (peaks = longer distances = troughs)
    long peakSum = 0;
    for (int i = 0; i < peakCount; i++) {
        peakSum += peaks[i];
    }
    int avgPeakDistance = peakSum / peakCount;  // Average trough distance
    
    // Calculate trough water level (longer distance = lower water level)
    *waterLevel = MOUNT_HEIGHT - avgPeakDistance;
    
    // Calculate average of lower envelope (troughs = shorter distances = crests)
    long troughSum = 0;
    for (int i = 0; i < troughCount; i++) {
        troughSum += troughs[i];
    }
    int avgTroughDistance = troughSum / troughCount;  // Average crest distance
    
    // Wave height = trough distance - crest distance (should be positive if logic is correct)
    int height = avgPeakDistance - avgTroughDistance;
    
    // Clean up allocated memory
    free(peaks);
    free(troughs);
    
    return height;
}

int calculateWaveHeightBinning(int samples[], int numSamples, int* waterLevel) {
    // Wave height calculation using binning analysis
    // This method bins distance readings and finds wave height as the difference between
    // minimum and maximum bins that contain at least a threshold number of samples
    // Input: samples[] contains distance values in mm (already converted)
    
    const int BIN_SIZE = 25;             // mm per bin (25mm = ~1 inch resolution)
    const int MIN_SAMPLES_PER_BIN = 8;   // Minimum samples required for a bin to be considered significant
    
    // Find the range of distance values
    int minDistance = samples[0];
    int maxDistance = samples[0];
    for (int i = 1; i < numSamples; i++) {
        if (samples[i] < minDistance) minDistance = samples[i];
        if (samples[i] > maxDistance) maxDistance = samples[i];
    }
    
    // Calculate number of bins needed
    int numBins = ((maxDistance - minDistance) / BIN_SIZE) + 1;
    
    // Create and initialize bin array - use heap allocation
    int* binCounts = (int*)malloc(numBins * sizeof(int));
    if (binCounts == NULL) {
        Serial.println("Error: Failed to allocate memory for binning calculation");
        *waterLevel = 0;
        return 0;
    }
    
    for (int i = 0; i < numBins; i++) {
        binCounts[i] = 0;
    }
    
    // Count samples in each bin
    for (int i = 0; i < numSamples; i++) {
        int binIndex = (samples[i] - minDistance) / BIN_SIZE;
        if (binIndex >= 0 && binIndex < numBins) {
            binCounts[binIndex]++;
        }
    }
    
    // Find the minimum and maximum bin indices that meet the threshold
    int minValidBin = -1;
    int maxValidBin = -1;
    
    for (int i = 0; i < numBins; i++) {
        if (binCounts[i] >= MIN_SAMPLES_PER_BIN) {
            if (minValidBin == -1) {
                minValidBin = i;
            }
            maxValidBin = i;
        }
    }
    
    // Return 0 if binning analysis failed to find sufficient bins
    if (minValidBin == -1 || maxValidBin == -1 || minValidBin == maxValidBin) {
        free(binCounts);
        *waterLevel = 0;
        return 0;
    }
    
    // Calculate distance values for the center of the min and max valid bins
    int minBinCenterDistance = minDistance + (minValidBin * BIN_SIZE) + (BIN_SIZE / 2);  // Shorter distances = crests
    int maxBinCenterDistance = minDistance + (maxValidBin * BIN_SIZE) + (BIN_SIZE / 2);  // Longer distances = troughs
    
    // Calculate trough water level (longer distance = lower water level)
    *waterLevel = MOUNT_HEIGHT - maxBinCenterDistance;
    
    // Wave height = trough distance - crest distance (should be positive if logic is correct)
    int height = maxBinCenterDistance - minBinCenterDistance;
    
    // Clean up allocated memory
    free(binCounts);
    
    return height;
}