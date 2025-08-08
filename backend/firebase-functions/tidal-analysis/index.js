/**
 * Matrix Pencil Tidal Analysis v1 for Tide Monitor
 * 
 * This Cloud Function performs advanced tidal harmonic analysis using the Matrix Pencil method.
 * The Matrix Pencil method decomposes signals into complex exponentials: f(t) = Σ Aₖ e^(σₖ + jωₖ)t
 * 
 * Features:
 * - Non-periodic signal analysis (handles real-world tidal variations)
 * - Automatic model order selection via SVD with 1% threshold
 * - Up to 8 signal components with conjugate pair filtering
 * - Comprehensive frequency, amplitude, damping, and phase estimation
 * - Scheduled execution every 5 minutes via Cloud Scheduler
 * 
 * Version: matrix-pencil-v1
 * Last Updated: 2025-08-01
 */

const { onCall } = require('firebase-functions/v2/https');
const { onSchedule } = require('firebase-functions/v2/scheduler');
const admin = require('firebase-admin');

// Initialize Firebase Admin
if (!admin.apps.length) {
    admin.initializeApp();
}

const db = admin.database();

// Configuration
const ANALYSIS_ENABLED = process.env.TIDAL_ANALYSIS_ENABLED === 'true';
const MATRIX_PENCIL_VERSION = 'matrix-pencil-v1';

/**
 * Matrix Pencil Method - Signal parameter estimation for non-periodic data
 * Estimates frequencies, damping, amplitudes, and phases from time series data
 */
function matrixPencilAnalysis(data, pencilParam = null) {
    const n = data.length;
    if (n < 20) return null; // Need sufficient data for Matrix Pencil
    
    const sn = data.length;
    const x = data.map(d => d.x);
    const y = data.map(d => d.y);
    
    console.log(`Matrix Pencil: Processing ${sn} samples`);
    
    // Calculate sampling parameters (assume regular intervals)
    const timeSpanMs = x[x.length - 1] - x[0];
    const timeSpanSec = timeSpanMs / 1000;
    const samplingInterval = timeSpanSec / (sn - 1);
    const startTime = x[0];
    
    // Remove DC component
    const mean = y.reduce((sum, val) => sum + val, 0) / sn;
    const yDetrended = y.map(val => val - mean);
    
    // Set pencil parameter L for proper tidal frequency resolution
    const L = Math.min(pencilParam || Math.floor(sn / 3), Math.floor(sn / 2)); // Use up to half the samples
    const M = sn - L + 1;
    
    console.log(`Matrix Pencil: L=${L}, M=${M}`);
    
    // Construct Hankel data matrix Y0 (M x L)
    const Y0 = [];
    for (let i = 0; i < M; i++) {
        const row = [];
        for (let j = 0; j < L; j++) {
            row.push(yDetrended[i + j]);
        }
        Y0.push(row);
    }
    
    // Construct shifted Hankel matrix Y1 (M x L)
    const Y1 = [];
    for (let i = 0; i < M - 1; i++) {
        const row = [];
        for (let j = 0; j < L; j++) {
            row.push(yDetrended[i + j + 1]);
        }
        Y1.push(row);
    }
    
    // SVD of Y0 to determine signal subspace
    console.log(`Matrix Pencil: Computing SVD of ${M}x${L} matrix`);
    let svd;
    svd = computeSVD(Y0);
    if (!svd || !svd.S || svd.S.length === 0) {
        throw new Error('SVD computation failed - no singular values returned');
    }
    console.log(`Matrix Pencil: SVD completed, ${svd.S.length} singular values`);
    
    // Determine model order using SVD threshold (keep components > 1% of max singular value)
    const threshold = 0.01 * Math.max(...svd.S);
    let modelOrder = 0;
    for (let i = 0; i < svd.S.length; i++) {
        if (svd.S[i] > threshold) {
            modelOrder++;
        } else {
            break;
        }
    }
    
    // Ensure we have at least 2 components for complex eigenvalue, but not too many
    modelOrder = Math.max(2, Math.min(modelOrder, 8));
    console.log(`Matrix Pencil: Determined model order = ${modelOrder} from ${svd.S.length} singular values`);
    console.log('Singular values:', svd.S);
    
    if (svd.S.length < modelOrder) {
        throw new Error(`Insufficient singular values: need ${modelOrder}, have ${svd.S.length}`);
    }
    
    // Extract signal subspace (first modelOrder columns of U and Vt)
    const Us = svd.U.slice(0, modelOrder);
    const Vs = svd.Vt.slice(0, modelOrder);
    
    // Form V1 (first L-1 columns) and V2 (last L-1 columns) - transpose of Vs submatrices
    const V1 = [];
    const V2 = [];
    
    // V1 and V2 should be (L-1) x modelOrder matrices
    for (let i = 0; i < L - 1; i++) {
        const v1Row = [];
        const v2Row = [];
        for (let j = 0; j < modelOrder; j++) {
            v1Row.push(Vs[j][i]);     // First L-1 elements
            v2Row.push(Vs[j][i + 1]); // Last L-1 elements (shifted by 1)
        }
        V1.push(v1Row);
        V2.push(v2Row);
    }
    
    console.log(`Matrix Pencil: V1 size = ${V1.length} x ${V1[0].length}, V2 size = ${V2.length} x ${V2[0].length}`);
    
    // Solve generalized eigenvalue problem: V1 * z = λ * V2 * z
    console.log(`Matrix Pencil: Computing eigenvalues`);
    let eigenvalues;
    eigenvalues = solveGeneralizedEigenvalue(V1, V2);
    if (!eigenvalues || eigenvalues.length === 0) {
        throw new Error('Eigenvalue computation failed - no eigenvalues returned');
    }
    console.log(`Matrix Pencil: Found ${eigenvalues.length} eigenvalues`);
    console.log('Eigenvalues:', eigenvalues);
    
    // Extract signal parameters from eigenvalues
    const signalComponents = [];
    
    // Process eigenvalues to detect tidal frequencies, keeping only positive frequencies
    for (let k = 0; k < eigenvalues.length; k++) {
        const lambda = eigenvalues[k];
        const magnitude = Math.sqrt((lambda.real || 0) * (lambda.real || 0) + (lambda.imag || 0) * (lambda.imag || 0));
        
        // Skip eigenvalues that are too small or exactly 1 (DC component)
        if (magnitude < 1e-6 || Math.abs(magnitude - 1.0) < 1e-6) {
            continue;
        }
        
        // Only process eigenvalues with positive imaginary part to avoid conjugate duplicates
        if ((lambda.imag || 0) <= 0) {
            continue;
        }
        
        // Convert eigenvalue to continuous-time parameters
        const lambdaMag = Math.sqrt((lambda.real || 0) * (lambda.real || 0) + (lambda.imag || 0) * (lambda.imag || 0));
        const s_real = Math.log(lambdaMag) / samplingInterval; // Damping (σ)
        const s_imag = Math.atan2(lambda.imag || 0, lambda.real || 0) / samplingInterval; // Frequency (ω)
        
        console.log(`Eigenvalue processing: lambdaMag=${lambdaMag}, s_real=${s_real}, s_imag=${s_imag}`);
        
        const frequency = Math.abs(s_imag);
        if (frequency > 1e-6) {
            const periodSec = (2 * Math.PI) / frequency;
            const periodMs = periodSec * 1000;
            const periodHours = periodMs / (60 * 60 * 1000);
            
            // Estimate amplitude and phase using least squares fitting
            const amplitude = estimateAmplitudePhase(x, yDetrended, s_real, s_imag, startTime, samplingInterval);
            const phase = amplitude ? amplitude.phase : 0;
            const amplitudeMag = amplitude ? amplitude.magnitude : 0;
            const dampingCoeff = s_real;
            
            console.log(`Component analysis: frequency=${frequency.toFixed(6)}, s_real=${s_real.toFixed(6)}, s_imag=${s_imag.toFixed(6)}, amplitudeMag=${amplitudeMag}`);
            
            signalComponents.push({
                frequency: frequency,         // rad/sec
                periodMs: periodMs,
                periodHours: periodHours,
                damping: dampingCoeff,        // damping coefficient (σ)
                amplitude: amplitudeMag,      // complex amplitude magnitude
                phase: phase,                 // phase in radians
                eigenvalue: lambda,           // original eigenvalue
                power: amplitudeMag * amplitudeMag  // power measure for sorting
            });
        }
    }
    
    // Sort by power (amplitude squared)
    signalComponents.sort((a, b) => b.power - a.power);
    
    // Output results to console
    console.log('Matrix Pencil Analysis Results:');
    console.log(`Model Order: ${modelOrder}, Pencil Parameter L: ${L}, Data Length: ${n}`);
    console.log(`Found ${signalComponents.length} signal components:`);
    signalComponents.forEach((comp, index) => {
        console.log(`Component ${index + 1}: Period=${comp.periodHours.toFixed(2)}h, Frequency=${comp.frequency.toFixed(6)} rad/s, Amplitude=${comp.amplitude.toFixed(4)}, Damping=${comp.damping.toFixed(6)}, Phase=${comp.phase.toFixed(4)} rad`);
    });
    
    return {
        components: signalComponents,
        dcComponent: mean,
        startTime: startTime,
        samplingInterval: samplingInterval,
        modelOrder: modelOrder,
        pencilParam: L,
        dataLength: n
    };
}

// SVD computation for Matrix Pencil
function computeSVD(matrix) {
    const m = matrix.length;
    const n = matrix[0].length;
    
    // Simple power iteration SVD for our use case
    const tolerance = 1e-6;
    const maxIterations = 50;
    
    const U = [];
    const S = [];
    const Vt = [];
    
    // Create copy of matrix for decomposition
    let A = matrix.map(row => [...row]);
    
    const rank = Math.min(m, n, 5); // Limit to first 5 singular values for speed
    
    for (let k = 0; k < rank; k++) {
        // Power iteration to find dominant singular value/vector
        let v = new Array(n).fill(0).map(() => Math.random() - 0.5);
        let prevSigma = 0;
        
        for (let iter = 0; iter < maxIterations; iter++) {
            // v = A^T * A * v
            const Av = new Array(m).fill(0);
            for (let i = 0; i < m; i++) {
                for (let j = 0; j < n; j++) {
                    Av[i] += A[i][j] * v[j];
                }
            }
            
            const AtAv = new Array(n).fill(0);
            for (let j = 0; j < n; j++) {
                for (let i = 0; i < m; i++) {
                    AtAv[j] += A[i][j] * Av[i];
                }
            }
            
            // Normalize
            const norm = Math.sqrt(AtAv.reduce((sum, val) => sum + val * val, 0));
            if (norm < tolerance) break;
            
            for (let j = 0; j < n; j++) {
                v[j] = AtAv[j] / norm;
            }
            
            // Calculate singular value
            const sigma = Math.sqrt(norm);
            
            if (Math.abs(sigma - prevSigma) < tolerance) break;
            prevSigma = sigma;
        }
        
        // Calculate u = A * v / sigma
        const Av = new Array(m).fill(0);
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                Av[i] += A[i][j] * v[j];
            }
        }
        
        const sigma = Math.sqrt(Av.reduce((sum, val) => sum + val * val, 0));
        if (sigma < tolerance) break;
        
        const u = Av.map(val => val / sigma);
        
        // Store singular vectors and value
        U.push(u);
        S.push(sigma);
        Vt.push([...v]);
        
        // Deflate matrix: A = A - σ * u * v^T
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                A[i][j] -= sigma * u[i] * v[j];
            }
        }
    }
    
    return { U, S, Vt };
}

// Matrix operations utilities
function matrixMultiply(A, B) {
    const m = A.length;
    const n = A[0].length;
    const p = B[0].length;
    
    if (n !== B.length) {
        throw new Error('Matrix dimensions incompatible for multiplication');
    }
    
    const result = [];
    for (let i = 0; i < m; i++) {
        result[i] = [];
        for (let j = 0; j < p; j++) {
            let sum = 0;
            for (let k = 0; k < n; k++) {
                sum += A[i][k] * B[k][j];
            }
            result[i][j] = sum;
        }
    }
    return result;
}

function matrixTranspose(A) {
    const m = A.length;
    const n = A[0].length;
    const result = [];
    
    for (let j = 0; j < n; j++) {
        result[j] = [];
        for (let i = 0; i < m; i++) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

function matrixInverse(A) {
    const n = A.length;
    
    // Create augmented matrix [A | I]
    const augmented = [];
    for (let i = 0; i < n; i++) {
        augmented[i] = [...A[i]];
        for (let j = 0; j < n; j++) {
            augmented[i][n + j] = (i === j) ? 1 : 0;
        }
    }
    
    // Gaussian elimination with partial pivoting
    for (let i = 0; i < n; i++) {
        // Find pivot
        let maxRow = i;
        for (let k = i + 1; k < n; k++) {
            if (Math.abs(augmented[k][i]) > Math.abs(augmented[maxRow][i])) {
                maxRow = k;
            }
        }
        
        // Swap rows
        [augmented[i], augmented[maxRow]] = [augmented[maxRow], augmented[i]];
        
        // Check for singular matrix
        if (Math.abs(augmented[i][i]) < 1e-12) {
            throw new Error(`Matrix is singular or near-singular at row ${i}`);
        }
        
        // Scale pivot row
        const pivot = augmented[i][i];
        for (let j = 0; j < 2 * n; j++) {
            augmented[i][j] /= pivot;
        }
        
        // Eliminate column
        for (let k = 0; k < n; k++) {
            if (k !== i) {
                const factor = augmented[k][i];
                for (let j = 0; j < 2 * n; j++) {
                    augmented[k][j] -= factor * augmented[i][j];
                }
            }
        }
    }
    
    // Extract inverse matrix
    const inverse = [];
    for (let i = 0; i < n; i++) {
        inverse[i] = augmented[i].slice(n);
    }
    
    return inverse;
}

// QR decomposition using Gram-Schmidt process
function qrDecomposition(A) {
    const m = A.length;
    const n = A[0].length;
    
    const Q = [];
    const R = [];
    
    // Initialize Q with zeros
    for (let i = 0; i < m; i++) {
        Q[i] = new Array(n).fill(0);
    }
    
    // Initialize R with zeros
    for (let i = 0; i < n; i++) {
        R[i] = new Array(n).fill(0);
    }
    
    // Gram-Schmidt process
    for (let j = 0; j < n; j++) {
        // Get column j of A
        const v = [];
        for (let i = 0; i < m; i++) {
            v[i] = A[i][j];
        }
        
        // Orthogonalize against previous columns
        for (let k = 0; k < j; k++) {
            // Compute R[k][j] = Q_k^T * v
            let dot = 0;
            for (let i = 0; i < m; i++) {
                dot += Q[i][k] * v[i];
            }
            R[k][j] = dot;
            
            // v = v - R[k][j] * Q_k
            for (let i = 0; i < m; i++) {
                v[i] -= R[k][j] * Q[i][k];
            }
        }
        
        // Compute R[j][j] = ||v||
        let norm = 0;
        for (let i = 0; i < m; i++) {
            norm += v[i] * v[i];
        }
        R[j][j] = Math.sqrt(norm);
        
        // Check for linear dependence
        if (R[j][j] < 1e-12) {
            throw new Error(`Matrix is rank deficient at column ${j}`);
        }
        
        // Q_j = v / ||v||
        for (let i = 0; i < m; i++) {
            Q[i][j] = v[i] / R[j][j];
        }
    }
    
    return { Q, R };
}

// Solve generalized eigenvalue problem for Matrix Pencil
function solveGeneralizedEigenvalue(V1, V2) {
    if (!V1 || !V2 || V1.length === 0 || V2.length === 0) return [];
    
    const m = V1.length;
    const n = V1[0].length;
    
    try {
        // Convert to standard eigenvalue problem: solve V1^+ * V2
        const V1T = matrixTranspose(V1);
        const V1TV1 = matrixMultiply(V1T, V1);
        
        // Check condition number
        const condition = checkConditionNumber(V1TV1);
        console.log('V1TV1 condition number:', condition);
        
        if (condition > 1e12) {
            throw new Error(`V1TV1 matrix is too ill-conditioned: condition number = ${condition}`);
        }
        
        const V1TV1_inv = matrixInverse(V1TV1);
        if (!V1TV1_inv) {
            throw new Error('Failed to invert V1^T * V1 - matrix is singular');
        }
        
        const V1_pinv = matrixMultiply(V1TV1_inv, V1T);
        const C = matrixMultiply(V1_pinv, V2);
        
        const eigenvalues = solveEigenvalues(C);
        return eigenvalues;
        
    } catch (error) {
        throw new Error(`Generalized eigenvalue computation failed: ${error.message}`);
    }
}

// QR algorithm for eigenvalue computation
function solveEigenvalues(A) {
    const n = A.length;
    const maxIterations = 20;
    const tolerance = 1e-4;
    
    let Ak = A.map(row => [...row]);
    
    for (let iter = 0; iter < maxIterations; iter++) {
        const qr = qrDecomposition(Ak);
        if (!qr) break;
        
        Ak = matrixMultiply(qr.R, qr.Q);
        
        // Check for convergence
        let offDiagNorm = 0;
        for (let i = 0; i < n; i++) {
            for (let j = 0; j < n; j++) {
                if (i !== j) {
                    offDiagNorm += Ak[i][j] * Ak[i][j];
                }
            }
        }
        
        if (Math.sqrt(offDiagNorm) < tolerance) {
            break;
        }
    }
    
    // Extract eigenvalues from diagonal
    const eigenvalues = [];
    for (let i = 0; i < n; i++) {
        if (i < n - 1 && Math.abs(Ak[i + 1][i]) > tolerance) {
            // 2x2 block - complex eigenvalues
            const a = Ak[i][i];
            const b = Ak[i][i + 1];
            const c = Ak[i + 1][i];
            const d = Ak[i + 1][i + 1];
            
            const trace = a + d;
            const det = a * d - b * c;
            const discriminant = trace * trace - 4 * det;
            
            if (discriminant < 0) {
                const realPart = trace / 2;
                const imagPart = Math.sqrt(-discriminant) / 2;
                
                eigenvalues.push({
                    real: realPart,
                    imag: imagPart
                });
                eigenvalues.push({
                    real: realPart,
                    imag: -imagPart
                });
                
                i++;
            } else {
                eigenvalues.push({
                    real: Ak[i][i],
                    imag: 0
                });
            }
        } else {
            eigenvalues.push({
                real: Ak[i][i],
                imag: 0
            });
        }
    }
    
    return eigenvalues;
}

// Check condition number
function checkConditionNumber(A) {
    const n = A.length;
    
    let maxDiag = 0;
    let minDiag = Infinity;
    
    for (let i = 0; i < n; i++) {
        const diag = Math.abs(A[i][i]);
        maxDiag = Math.max(maxDiag, diag);
        minDiag = Math.min(minDiag, diag);
    }
    
    if (minDiag === 0) return Infinity;
    return maxDiag / minDiag;
}

// Estimate amplitude and phase
function estimateAmplitudePhase(timeValues, dataValues, damping, frequency, startTime, samplingInterval) {
    const n = dataValues.length;
    
    if (!isFinite(damping) || !isFinite(frequency) || frequency === 0) {
        console.log('Invalid parameters for amplitude estimation: damping=', damping, 'frequency=', frequency);
        return null;
    }
    
    // Build design matrix for least squares
    const A = [];
    const b = [];
    
    for (let i = 0; i < n; i++) {
        const t = i * samplingInterval;
        const expDamping = Math.exp(damping * t);
        
        if (!isFinite(expDamping)) {
            console.log('Exponential damping overflow at t=', t, 'damping=', damping);
            return null;
        }
        
        const cosComponent = Math.cos(frequency * t) * expDamping;
        const sinComponent = Math.sin(frequency * t) * expDamping;
        
        A.push([cosComponent, sinComponent]);
        b.push(dataValues[i]);
    }
    
    try {
        const AT = matrixTranspose(A);
        const ATA = matrixMultiply(AT, A);
        const ATA_inv = matrixInverse(ATA);
        
        if (!ATA_inv) {
            throw new Error('Matrix inversion failed in amplitude estimation');
        }
        
        const ATb = [];
        for (let i = 0; i < 2; i++) {
            let sum = 0;
            for (let j = 0; j < n; j++) {
                sum += AT[i][j] * b[j];
            }
            ATb.push(sum);
        }
        
        const coeffs = [];
        for (let i = 0; i < 2; i++) {
            let sum = 0;
            for (let j = 0; j < 2; j++) {
                sum += ATA_inv[i][j] * ATb[j];
            }
            coeffs.push(sum);
        }
        
        const a = coeffs[0];
        const b_coeff = coeffs[1];
        
        const magnitude = Math.sqrt(a * a + b_coeff * b_coeff);
        const phase = Math.atan2(-b_coeff, a);
        
        return { magnitude, phase };
        
    } catch (error) {
        throw new Error(`Amplitude/phase estimation failed: ${error.message}`);
    }
}

/**
 * Scheduled function to run Matrix Pencil analysis every 5 minutes
 */
exports.runTidalAnalysis = onSchedule('*/5 * * * *', async (event) => {
    if (!ANALYSIS_ENABLED) {
        console.log('Tidal analysis is disabled. Set TIDAL_ANALYSIS_ENABLED=true to enable.');
        return;
    }

    const startTime = Date.now();
    console.log('Starting Matrix Pencil tidal analysis...');

    try {
        // Fetch the last 72 hours of readings (4320 samples at 1-minute intervals)
        const readingsRef = db.ref('readings');
        const snapshot = await readingsRef.orderByKey().limitToLast(4320).once('value');
        const readings = snapshot.val();

        if (!readings) {
            throw new Error('No readings data available');
        }

        // Convert to array format expected by Matrix Pencil
        const dataPoints = Object.entries(readings)
            .map(([key, value]) => ({
                x: new Date(value.t).getTime(),
                y: value.w / 304.8  // Convert mm to feet
            }))
            .filter(point => !isNaN(point.y) && isFinite(point.y))
            .sort((a, b) => a.x - b.x);

        if (dataPoints.length < 100) {
            throw new Error(`Insufficient data points: ${dataPoints.length} (need at least 100)`);
        }

        console.log(`Processing ${dataPoints.length} data points over ${((dataPoints[dataPoints.length - 1].x - dataPoints[0].x) / (1000 * 60 * 60)).toFixed(1)} hours`);

        // Run Matrix Pencil analysis
        const analysisResult = matrixPencilAnalysis(dataPoints);

        if (!analysisResult || !analysisResult.components) {
            throw new Error('Matrix Pencil analysis returned no results');
        }

        // Prepare results for storage
        const results = {
            methodology: MATRIX_PENCIL_VERSION,
            timestamp: new Date().toISOString(),
            computationTimeMs: Date.now() - startTime,
            dataPoints: dataPoints.length,
            timeSpanHours: ((dataPoints[dataPoints.length - 1].x - dataPoints[0].x) / (1000 * 60 * 60)).toFixed(1),
            components: analysisResult.components.map(comp => ({
                frequency: comp.frequency,
                periodMs: comp.periodMs,
                periodHours: comp.periodHours,
                damping: comp.damping,
                amplitude: comp.amplitude,
                phase: comp.phase,
                power: comp.power
            })),
            dcComponent: analysisResult.dcComponent,
            modelOrder: analysisResult.modelOrder,
            pencilParam: analysisResult.pencilParam
        };

        // Store results in Firebase
        await db.ref('tidal-analysis').set(results);

        console.log(`Matrix Pencil analysis completed in ${results.computationTimeMs}ms`);
        console.log(`Found ${results.components.length} tidal components`);
        results.components.forEach((comp, i) => {
            console.log(`Component ${i + 1}: ${comp.periodHours.toFixed(2)}h period, ${comp.amplitude.toFixed(4)} amplitude`);
        });

    } catch (error) {
        console.error('Tidal analysis failed:', error.message);
        console.error('Stack trace:', error.stack);

        // Clear old results on failure to indicate problem
        await db.ref('tidal-analysis').remove();

        // Store error information
        await db.ref('tidal-analysis-error').set({
            timestamp: new Date().toISOString(),
            error: error.message,
            methodology: MATRIX_PENCIL_VERSION
        });

        throw error;
    }
});

/**
 * HTTP callable function for manual analysis trigger (for testing)
 */
exports.triggerTidalAnalysis = onCall(async (request) => {
    console.log('Manual tidal analysis trigger received');
    
    // Temporarily enable analysis for manual trigger
    const wasEnabled = ANALYSIS_ENABLED;
    process.env.TIDAL_ANALYSIS_ENABLED = 'true';
    
    try {
        await exports.runTidalAnalysis.run();
        return { success: true, message: 'Analysis completed successfully' };
    } catch (error) {
        return { success: false, error: error.message };
    } finally {
        // Restore original setting
        process.env.TIDAL_ANALYSIS_ENABLED = wasEnabled.toString();
    }
});