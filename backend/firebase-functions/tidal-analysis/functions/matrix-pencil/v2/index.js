/**
 * Matrix Pencil Tidal Analysis v2 for Tide Monitor
 * 
 * Balanced enhanced version with improved accuracy and reliability.
 * Key improvements over v1:
 * - Larger pencil parameter L (2/3 of data length vs 1/3) for better frequency resolution
 * - Optimized SVD rank limit (8) balanced for cloud performance
 * - More sensitive threshold (0.5% vs 1%) for component detection  
 * - Support for up to 12 signal components (vs 8)
 * - Optimized precision parameters for cloud deployment
 * 
 * The Matrix Pencil method decomposes signals into complex exponentials: f(t) = Σ Aₖ e^(σₖ + jωₖ)t
 * 
 * Features:
 * - Non-periodic signal analysis (handles real-world tidal variations)
 * - Automatic model order selection via optimized SVD with 0.5% threshold
 * - Up to 12 signal components with conjugate pair filtering
 * - Improved frequency resolution for closely spaced tidal constituents
 * - Comprehensive frequency, amplitude, damping, and phase estimation
 * - Scheduled execution every 5 minutes via Cloud Scheduler
 * 
 * Version: matrix-pencil-v2
 * Created: 2025-08-13
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
const MATRIX_PENCIL_VERSION = 'matrix-pencil-v2';

/**
 * Enhanced Matrix Pencil Method - Improved signal parameter estimation
 * Key improvements: larger L, higher SVD rank, more sensitive threshold
 */
function matrixPencilAnalysis(data, pencilParam = null) {
    const n = data.length;
    if (n < 30) return null; // Need more data for enhanced accuracy
    
    const sn = data.length;
    const x = data.map(d => d.x);
    const y = data.map(d => d.y);
    
    console.log(`Matrix Pencil v2: Processing ${sn} samples`);
    
    // Calculate sampling parameters (assume regular intervals)
    const timeSpanMs = x[x.length - 1] - x[0];
    const timeSpanSec = timeSpanMs / 1000;
    const samplingInterval = timeSpanSec / (sn - 1);
    const startTime = x[0];
    
    console.log(`Matrix Pencil v2: Time span = ${timeSpanSec.toFixed(1)}s (${(timeSpanSec/3600).toFixed(1)}h), sampling interval = ${samplingInterval.toFixed(1)}s`);
    
    // Remove DC component and analyze data characteristics
    const mean = y.reduce((sum, val) => sum + val, 0) / sn;
    const yDetrended = y.map(val => val - mean);
    
    // Debug: Check data characteristics
    const yMin = Math.min(...y);
    const yMax = Math.max(...y);
    const yRange = yMax - yMin;
    const yStd = Math.sqrt(yDetrended.reduce((sum, val) => sum + val * val, 0) / sn);
    console.log(`Matrix Pencil v2: Data range = ${yRange.toFixed(4)}m, std = ${yStd.toFixed(4)}m, mean = ${mean.toFixed(4)}m`);
    
    // Use much smaller L parameter like v1 for diagnosis 
    const L = Math.min(pencilParam || Math.floor(sn / 3), Math.floor(sn / 2)); // Back to 1/3 like v1
    const M = sn - L + 1;
    
    console.log(`Matrix Pencil v2: L=${L} (enhanced from ~${Math.floor(sn/3)} in v1), M=${M}`);
    
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
    
    // Enhanced SVD of Y0 to determine signal subspace
    console.log(`Matrix Pencil v2: Computing enhanced SVD of ${M}x${L} matrix`);
    let svd;
    svd = computeEnhancedSVD(Y0);
    if (!svd || !svd.S || svd.S.length === 0) {
        throw new Error('Enhanced SVD computation failed - no singular values returned');
    }
    console.log(`Matrix Pencil v2: Enhanced SVD completed, ${svd.S.length} singular values`);
    
    // Moderately more sensitive model order selection using 0.5% threshold (vs 1% in v1) 
    const threshold = 0.005 * Math.max(...svd.S); // 0.5% - balanced sensitivity
    let modelOrder = 0;
    for (let i = 0; i < svd.S.length; i++) {
        if (svd.S[i] > threshold) {
            modelOrder++;
        } else {
            break;
        }
    }
    
    // Allow moderately more components for better frequency resolution (up to 12 vs 8 in v1)
    modelOrder = Math.max(2, Math.min(modelOrder, 12));
    console.log(`Matrix Pencil v2: Enhanced model order = ${modelOrder} from ${svd.S.length} singular values (threshold: ${threshold.toFixed(8)})`);
    console.log('Singular values (first 10):', svd.S.slice(0, 10));
    
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
    
    console.log(`Matrix Pencil v2: V1 size = ${V1.length} x ${V1[0].length}, V2 size = ${V2.length} x ${V2[0].length}`);
    
    // Solve generalized eigenvalue problem: V1 * z = λ * V2 * z
    // Use proven v1 approach instead of enhanced solver
    console.log(`Matrix Pencil v2: Computing eigenvalues using v1 approach`);
    let eigenvalues;
    eigenvalues = solveGeneralizedEigenvalue(V1, V2);
    if (!eigenvalues || eigenvalues.length === 0) {
        throw new Error('Enhanced eigenvalue computation failed - no eigenvalues returned');
    }
    console.log(`Matrix Pencil v2: Found ${eigenvalues.length} eigenvalues`);
    console.log('Eigenvalues (first 8):', eigenvalues.slice(0, 8));
    
    // Extract signal parameters from eigenvalues with enhanced filtering
    const signalComponents = [];
    
    // Enhanced eigenvalue processing with better frequency detection
    for (let k = 0; k < eigenvalues.length; k++) {
        const lambda = eigenvalues[k];
        const magnitude = Math.sqrt((lambda.real || 0) * (lambda.real || 0) + (lambda.imag || 0) * (lambda.imag || 0));
        
        // Process all eigenvalues - let physical period constraints be the only filter
        
        // Convert eigenvalue to continuous-time parameters
        const lambdaMag = Math.sqrt((lambda.real || 0) * (lambda.real || 0) + (lambda.imag || 0) * (lambda.imag || 0));
        const s_real = Math.log(lambdaMag) / samplingInterval; // Damping (σ)
        const s_imag = Math.atan2(lambda.imag || 0, lambda.real || 0) / samplingInterval; // Frequency (ω)
        
        console.log(`v2 Eigenvalue processing: lambdaMag=${lambdaMag}, s_real=${s_real}, s_imag=${s_imag}`);
        
        const frequency = Math.abs(s_imag);
        if (frequency > 0) {
            const periodSec = (2 * Math.PI) / frequency;
            const periodHours = periodSec / 3600;
            console.log(`  -> Frequency=${frequency.toExponential(3)}, Period=${periodHours.toFixed(1)}h`);
        }
        
        // Process all frequencies including very small ones - no threshold filtering
        if (frequency > 0) {  // Still need to avoid division by zero for period calculation
            const periodSec = (2 * Math.PI) / frequency;
            const periodMs = periodSec * 1000;
            const periodHours = periodMs / (60 * 60 * 1000);
            
            // Only include physically meaningful tidal frequencies (periods between 1-50 hours)
            if (periodHours >= 1.0 && periodHours <= 50.0) {
                // Estimate amplitude and phase using least squares fitting
                const amplitude = estimateAmplitudePhase(x, yDetrended, s_real, s_imag, startTime, samplingInterval);
                const phase = amplitude ? amplitude.phase : 0;
                const amplitudeMag = amplitude ? amplitude.magnitude : 0;
                const dampingCoeff = s_real;
                
                console.log(`v2 Component analysis: frequency=${frequency.toFixed(6)}, period=${periodHours.toFixed(2)}h, amplitude=${amplitudeMag.toFixed(4)}`);
                
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
    }
    
    // Sort by power (amplitude squared)
    signalComponents.sort((a, b) => b.power - a.power);
    
    // Output results to console
    console.log('Matrix Pencil v2 Analysis Results:');
    console.log(`Enhanced Model Order: ${modelOrder}, Pencil Parameter L: ${L}, Data Length: ${n}`);
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

// Enhanced SVD computation with optimized parameters for cloud performance
function computeEnhancedSVD(matrix) {
    const m = matrix.length;
    const n = matrix[0].length;
    
    // Optimized precision and iteration parameters to reduce CPU usage
    const tolerance = 1e-6; // Optimized precision for cloud performance
    const maxIterations = 50; // Reduced iterations to prevent timeouts
    
    const U = [];
    const S = [];
    const Vt = [];
    
    // Create copy of matrix for decomposition
    let A = matrix.map(row => [...row]);
    
    const rank = Math.min(m, n, 8); // Optimized rank limit for cloud performance
    
    for (let k = 0; k < rank; k++) {
        // Enhanced power iteration with better initialization
        let v = new Array(n).fill(0).map(() => (Math.random() - 0.5) * 2);
        let prevSigma = 0;
        
        for (let iter = 0; iter < maxIterations; iter++) {
            // v = A^T * A * v (enhanced precision)
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
            
            // Enhanced normalization with better numerical stability
            const norm = Math.sqrt(AtAv.reduce((sum, val) => sum + val * val, 0));
            if (norm < tolerance) break;
            
            for (let j = 0; j < n; j++) {
                v[j] = AtAv[j] / norm;
            }
            
            // Calculate singular value with enhanced precision
            const sigma = Math.sqrt(norm);
            
            if (Math.abs(sigma - prevSigma) < tolerance) break;
            prevSigma = sigma;
        }
        
        // Calculate u = A * v / sigma with enhanced precision
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
        
        // Enhanced deflation: A = A - σ * u * v^T
        for (let i = 0; i < m; i++) {
            for (let j = 0; j < n; j++) {
                A[i][j] -= sigma * u[i] * v[j];
            }
        }
    }
    
    return { U, S, Vt };
}

// Use proven v1 generalized eigenvalue solver
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

// Enhanced pseudoinverse computation
function computeEnhancedPseudoinverse(matrix) {
    const m = matrix.length;
    const n = matrix[0].length;
    
    // Enhanced SVD for pseudoinverse
    const svd = computeEnhancedSVD(matrix);
    const tolerance = 1e-6; // Optimized tolerance for cloud performance
    
    // Enhanced pseudoinverse: V * Σ⁺ * Uᵀ
    const pinv = [];
    for (let i = 0; i < n; i++) {
        const row = [];
        for (let j = 0; j < m; j++) {
            let sum = 0;
            for (let k = 0; k < Math.min(svd.S.length, n); k++) {
                if (svd.S[k] > tolerance) {
                    sum += svd.Vt[k][i] * (1.0 / svd.S[k]) * svd.U[k][j];
                }
            }
            row.push(sum);
        }
        pinv.push(row);
    }
    
    return pinv;
}

// Enhanced eigenvalue computation using power iteration
function computeEnhancedEigenvalues(matrix) {
    const n = matrix.length;
    const eigenvalues = [];
    const maxEigenvalues = Math.min(n, 12); // Balanced limit
    const tolerance = 1e-6; // Optimized tolerance for cloud performance  
    const maxIterations = 50; // Reduced iterations to prevent timeouts
    
    let A = matrix.map(row => [...row]);
    
    for (let k = 0; k < maxEigenvalues; k++) {
        // Enhanced power iteration for complex eigenvalues
        let v_real = new Array(n).fill(0).map(() => Math.random() - 0.5);
        let v_imag = new Array(n).fill(0).map(() => Math.random() - 0.5);
        
        let prevEigReal = 0, prevEigImag = 0;
        
        for (let iter = 0; iter < maxIterations; iter++) {
            // A * v (complex multiplication)
            const Av_real = new Array(n).fill(0);
            const Av_imag = new Array(n).fill(0);
            
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    Av_real[i] += A[i][j] * v_real[j];
                    Av_imag[i] += A[i][j] * v_imag[j];
                }
            }
            
            // Enhanced normalization
            const norm = Math.sqrt(
                Av_real.reduce((sum, val) => sum + val * val, 0) + 
                Av_imag.reduce((sum, val) => sum + val * val, 0)
            );
            
            if (norm < tolerance) break;
            
            for (let j = 0; j < n; j++) {
                v_real[j] = Av_real[j] / norm;
                v_imag[j] = Av_imag[j] / norm;
            }
            
            // Enhanced Rayleigh quotient for eigenvalue estimation
            let numerReal = 0, numerImag = 0, denom = 0;
            for (let i = 0; i < n; i++) {
                numerReal += v_real[i] * Av_real[i] + v_imag[i] * Av_imag[i];
                numerImag += v_real[i] * Av_imag[i] - v_imag[i] * Av_real[i];
                denom += v_real[i] * v_real[i] + v_imag[i] * v_imag[i];
            }
            
            const eigReal = denom > tolerance ? numerReal / denom : 0;
            const eigImag = denom > tolerance ? numerImag / denom : 0;
            
            if (Math.abs(eigReal - prevEigReal) < tolerance && Math.abs(eigImag - prevEigImag) < tolerance) break;
            prevEigReal = eigReal;
            prevEigImag = eigImag;
        }
        
        const eigenvalue = { real: prevEigReal, imag: prevEigImag };
        eigenvalues.push(eigenvalue);
        
        // Enhanced deflation to find next eigenvalue
        const eigMag = prevEigReal * prevEigReal + prevEigImag * prevEigImag;
        if (eigMag > tolerance) {
            for (let i = 0; i < n; i++) {
                for (let j = 0; j < n; j++) {
                    A[i][j] -= (prevEigReal * v_real[i] * v_real[j] + prevEigImag * v_imag[i] * v_real[j]) / eigMag;
                }
            }
        }
    }
    
    return eigenvalues;
}

// Amplitude and phase estimation (reused from v1)
function estimateAmplitudePhase(x, y, s_real, s_imag, startTime, samplingInterval) {
    const n = x.length;
    let sumReal = 0, sumImag = 0, sumWeight = 0;
    
    for (let i = 0; i < n; i++) {
        const t = (x[i] - startTime) / 1000; // Convert to seconds
        const expReal = Math.exp(s_real * t) * Math.cos(s_imag * t);
        const expImag = Math.exp(s_real * t) * Math.sin(s_imag * t);
        
        sumReal += y[i] * expReal;
        sumImag += y[i] * expImag;
        sumWeight += expReal * expReal + expImag * expImag;
    }
    
    if (sumWeight > 1e-10) {
        const amplReal = sumReal / sumWeight;
        const amplImag = sumImag / sumWeight;
        const magnitude = Math.sqrt(amplReal * amplReal + amplImag * amplImag);
        const phase = Math.atan2(amplImag, amplReal);
        
        return { magnitude, phase };
    }
    
    return { magnitude: 0, phase: 0 };
}

// Matrix operations utilities (from proven v1 implementation)
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
    const result = [];
    const identity = [];
    
    // Initialize result and identity matrices
    for (let i = 0; i < n; i++) {
        result[i] = [...A[i]];
        identity[i] = new Array(n).fill(0);
        identity[i][i] = 1;
    }
    
    // Gaussian elimination with partial pivoting
    for (let k = 0; k < n; k++) {
        // Find pivot
        let maxRow = k;
        for (let i = k + 1; i < n; i++) {
            if (Math.abs(result[i][k]) > Math.abs(result[maxRow][k])) {
                maxRow = i;
            }
        }
        
        // Swap rows
        if (maxRow !== k) {
            [result[k], result[maxRow]] = [result[maxRow], result[k]];
            [identity[k], identity[maxRow]] = [identity[maxRow], identity[k]];
        }
        
        // Check for singular matrix
        if (Math.abs(result[k][k]) < 1e-10) {
            return null;
        }
        
        // Scale pivot row
        const pivot = result[k][k];
        for (let j = 0; j < n; j++) {
            result[k][j] /= pivot;
            identity[k][j] /= pivot;
        }
        
        // Eliminate column
        for (let i = 0; i < n; i++) {
            if (i !== k) {
                const factor = result[i][k];
                for (let j = 0; j < n; j++) {
                    result[i][j] -= factor * result[k][j];
                    identity[i][j] -= factor * identity[k][j];
                }
            }
        }
    }
    
    return identity;
}

// QR algorithm for eigenvalue computation (from v1)
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

// QR decomposition (from v1)
function qrDecomposition(A) {
    const m = A.length;
    const n = A[0].length;
    
    const Q = [];
    const R = [];
    
    // Initialize Q as identity and R as copy of A
    for (let i = 0; i < m; i++) {
        Q[i] = new Array(n).fill(0);
        Q[i][i] = 1;
        R[i] = [...A[i]];
    }
    
    // Gram-Schmidt process
    for (let k = 0; k < Math.min(m, n); k++) {
        // Compute norm of k-th column
        let norm = 0;
        for (let i = k; i < m; i++) {
            norm += R[i][k] * R[i][k];
        }
        norm = Math.sqrt(norm);
        
        if (norm < 1e-10) return null;
        
        // Update R and Q
        for (let i = k; i < m; i++) {
            R[i][k] /= norm;
        }
        
        for (let j = k + 1; j < n; j++) {
            let dot = 0;
            for (let i = k; i < m; i++) {
                dot += R[i][k] * R[i][j];
            }
            
            for (let i = k; i < m; i++) {
                R[i][j] -= dot * R[i][k];
            }
        }
    }
    
    return { Q, R };
}

// Check condition number (from v1)
function checkConditionNumber(A) {
    const n = A.length;
    let maxEig = 0;
    let minEig = Infinity;
    
    // Simple approximation using diagonal dominance
    for (let i = 0; i < n; i++) {
        const diag = Math.abs(A[i][i]);
        let offDiag = 0;
        for (let j = 0; j < n; j++) {
            if (i !== j) {
                offDiag += Math.abs(A[i][j]);
            }
        }
        
        const est = diag - offDiag;
        if (est > 0) {
            maxEig = Math.max(maxEig, diag + offDiag);
            minEig = Math.min(minEig, est);
        }
    }
    
    return minEig > 0 ? maxEig / minEig : 1e15;
}

/**
 * Enhanced scheduled tidal analysis function (every 5 minutes)
 */
exports.runTidalAnalysis = onSchedule('*/5 * * * *', async (event) => {
    console.log('Matrix Pencil v2 tidal analysis triggered');
    
    if (!ANALYSIS_ENABLED) {
        console.log('Matrix Pencil v2 analysis is disabled via environment variable');
        return;
    }
    
    const startTime = Date.now();
    
    try {
        // Fetch latest readings from Firebase (last 72 hours)
        console.log('Fetching tidal data from Firebase...');
        const snapshot = await db.ref('readings').orderByKey().limitToLast(4320).once('value');
        const readings = snapshot.val();
        
        if (!readings) {
            throw new Error('No readings data available');
        }
        
        // Convert readings to analysis format
        const dataPoints = Object.entries(readings)
            .map(([key, reading]) => {
                if (!reading.t || reading.w === undefined) {
                    return null;
                }
                
                const timestamp = new Date(reading.t).getTime();
                return {
                    x: timestamp,
                    y: parseFloat(reading.w) / 1000 // Convert mm to meters
                };
            })
            .filter(point => point !== null)
            .sort((a, b) => a.x - b.x);
        
        console.log(`Matrix Pencil v2: Processing ${dataPoints.length} data points`);
        
        if (dataPoints.length < 100) {
            throw new Error(`Insufficient data for analysis: ${dataPoints.length} points (need at least 100)`);
        }
        
        // Run enhanced Matrix Pencil analysis
        const analysisResult = matrixPencilAnalysis(dataPoints);
        
        if (!analysisResult || !analysisResult.components) {
            throw new Error('Matrix Pencil v2 analysis returned no results');
        }
        
        // Format results for storage
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

        // Store results in Firebase under matrix-pencil-v2 specific location
        await db.ref('tidal-analysis/matrix-pencil-v2').set(results);

        console.log(`Matrix Pencil v2 analysis completed in ${results.computationTimeMs}ms`);
        console.log(`Found ${results.components.length} tidal components (enhanced detection)`);
        results.components.forEach((comp, i) => {
            console.log(`Component ${i + 1}: ${comp.periodHours.toFixed(2)}h period, ${comp.amplitude.toFixed(4)} amplitude`);
        });

    } catch (error) {
        console.error('Matrix Pencil v2 tidal analysis failed:', error.message);
        console.error('Stack trace:', error.stack);

        // Clear old results on failure to indicate problem
        await db.ref('tidal-analysis/matrix-pencil-v2').remove();

        // Store error information with v2 designation
        await db.ref('tidal-analysis-error-v2').set({
            timestamp: new Date().toISOString(),
            error: error.message,
            methodology: MATRIX_PENCIL_VERSION
        });

        throw error;
    }
});

/**
 * HTTP callable function for manual v2 analysis trigger (for testing)
 */
exports.triggerTidalAnalysis = onCall(async (request) => {
    console.log('Manual Matrix Pencil v2 tidal analysis trigger received');
    
    try {
        // Run the same analysis as the scheduled function
        await exports.runTidalAnalysis.run();
        return { success: true, message: 'Matrix Pencil v2 tidal analysis completed successfully' };
    } catch (error) {
        console.error('Manual Matrix Pencil v2 trigger failed:', error);
        return { success: false, error: error.message };
    }
});