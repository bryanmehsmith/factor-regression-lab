// Small dense matrix helpers. Matrices here are at most ~7x7 (an intercept
// plus up to 6 factors), so plain Gauss-Jordan inversion carries no numerical
// stability concerns at this scale, no need for QR or SVD.
//
// A matrix is number[][] (rows of rows); a vector is number[].

export function transpose(matrix) {
  const rows = matrix.length;
  const cols = matrix[0].length;
  const result = Array.from({ length: cols }, () => new Array(rows));
  for (let i = 0; i < rows; i++) {
    for (let j = 0; j < cols; j++) result[j][i] = matrix[i][j];
  }
  return result;
}

export function multiply(a, b) {
  const rows = a.length;
  const inner = b.length;
  const cols = b[0].length;
  const result = Array.from({ length: rows }, () => new Array(cols).fill(0));
  for (let i = 0; i < rows; i++) {
    for (let k = 0; k < inner; k++) {
      const aik = a[i][k];
      if (aik === 0) continue;
      for (let j = 0; j < cols; j++) result[i][j] += aik * b[k][j];
    }
  }
  return result;
}

export function multiplyVector(matrix, vector) {
  return matrix.map((row) => row.reduce((sum, value, j) => sum + value * vector[j], 0));
}

export function subtractVectors(a, b) {
  return a.map((value, i) => value - b[i]);
}

// Gauss-Jordan inversion with partial pivoting.
export function inverse(matrix) {
  const n = matrix.length;
  const augmented = matrix.map((row, i) => [
    ...row,
    ...Array.from({ length: n }, (_, j) => (i === j ? 1 : 0)),
  ]);

  for (let col = 0; col < n; col++) {
    let pivotRow = col;
    for (let row = col + 1; row < n; row++) {
      if (Math.abs(augmented[row][col]) > Math.abs(augmented[pivotRow][col])) pivotRow = row;
    }
    if (pivotRow !== col) [augmented[col], augmented[pivotRow]] = [augmented[pivotRow], augmented[col]];

    const pivot = augmented[col][col];
    if (Math.abs(pivot) < 1e-14) throw new Error("matrix is singular or near-singular");

    for (let j = 0; j < 2 * n; j++) augmented[col][j] /= pivot;
    for (let row = 0; row < n; row++) {
      if (row === col) continue;
      const factor = augmented[row][col];
      if (factor === 0) continue;
      for (let j = 0; j < 2 * n; j++) augmented[row][j] -= factor * augmented[col][j];
    }
  }

  return augmented.map((row) => row.slice(n));
}

// Column-vector-of-vectors helper: builds [1, x1, x2, ...] design rows.
export function designMatrix(columns) {
  const n = columns[0].length;
  const rows = [];
  for (let i = 0; i < n; i++) rows.push([1, ...columns.map((col) => col[i])]);
  return rows;
}
