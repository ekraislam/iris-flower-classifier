/**
 * Iris Flower Species Classifier - Frontend Logic
 * Communicates with Vercel Serverless API (/api/predict)
 */

document.addEventListener('DOMContentLoaded', () => {
  // DOM Elements
  const form = document.getElementById('prediction-form');
  const predictBtn = document.getElementById('predict-btn');
  const resetBtn = document.getElementById('reset-btn');
  const errorBox = document.getElementById('error-box');
  const errorMessage = document.getElementById('error-message');

  const resultPlaceholder = document.getElementById('result-placeholder');
  const resultContent = document.getElementById('result-content');
  const resultBanner = document.getElementById('result-banner');
  const resultIcon = document.getElementById('result-icon');
  const resultSpecies = document.getElementById('result-species');
  const resultScientific = document.getElementById('result-scientific');
  const resultDesc = document.getElementById('result-desc');
  const confidenceVal = document.getElementById('confidence-val');
  const confidenceBar = document.getElementById('confidence-bar');
  const probList = document.getElementById('prob-list');

  const sepalLengthInput = document.getElementById('sepal_length');
  const sepalWidthInput = document.getElementById('sepal_width');
  const petalLengthInput = document.getElementById('petal_length');
  const petalWidthInput = document.getElementById('petal_width');

  // Quick Preset Handlers
  const presetChips = document.querySelectorAll('.preset-chip');
  presetChips.forEach(chip => {
    chip.addEventListener('click', () => {
      sepalLengthInput.value = chip.dataset.sl;
      sepalWidthInput.value = chip.dataset.sw;
      petalLengthInput.value = chip.dataset.pl;
      petalWidthInput.value = chip.dataset.pw;

      hideError();

      // Brief animation feedback on inputs
      [sepalLengthInput, sepalWidthInput, petalLengthInput, petalWidthInput].forEach(input => {
        input.style.borderColor = 'var(--primary-accent)';
        setTimeout(() => {
          input.style.borderColor = '';
        }, 300);
      });
    });
  });

  // Reset Form
  resetBtn.addEventListener('click', () => {
    form.reset();
    hideError();
    resultPlaceholder.classList.remove('hidden');
    resultContent.classList.add('hidden');
    sepalLengthInput.focus();
  });

  // Display Error State
  function showError(msg) {
    errorMessage.textContent = msg;
    errorBox.classList.remove('hidden');
    predictBtn.classList.remove('btn-loading');
    predictBtn.disabled = false;
  }

  // Clear Error State
  function hideError() {
    errorMessage.textContent = '';
    errorBox.classList.add('hidden');
  }

  // Handle Predict Submission
  form.addEventListener('submit', async (e) => {
    e.preventDefault();
    hideError();

    const slStr = sepalLengthInput.value.trim();
    const swStr = sepalWidthInput.value.trim();
    const plStr = petalLengthInput.value.trim();
    const pwStr = petalWidthInput.value.trim();

    // Check empty fields
    if (!slStr || !swStr || !plStr || !pwStr) {
      showError('Please enter values for all four measurements.');
      return;
    }

    const sl = parseFloat(slStr);
    const sw = parseFloat(swStr);
    const pl = parseFloat(plStr);
    const pw = parseFloat(pwStr);

    // Validate numbers
    if (isNaN(sl) || isNaN(sw) || isNaN(pl) || isNaN(pw)) {
      showError('All measurements must be valid numeric values.');
      return;
    }

    if (sl <= 0 || sw <= 0 || pl <= 0 || pw <= 0) {
      showError('Measurements must be greater than 0 cm.');
      return;
    }

    if (sl > 30 || sw > 30 || pl > 30 || pw > 30) {
      showError('Measurements must be reasonable flower dimensions (≤ 30 cm).');
      return;
    }

    // Set Loading State
    predictBtn.classList.add('btn-loading');
    predictBtn.disabled = true;

    try {
      // Call Vercel Serverless Function endpoint
      const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Accept': 'application/json'
        },
        body: JSON.stringify({
          sepal_length: sl,
          sepal_width: sw,
          petal_length: pl,
          petal_width: pw
        })
      });

      const data = await response.json();

      if (!response.ok || !data.success) {
        showError(data.error || 'Prediction request failed. Please check inputs.');
        return;
      }

      displayResult(data.result);
    } catch (err) {
      console.error('API Error:', err);
      showError('Could not connect to prediction API. Make sure the server is running.');
    } finally {
      predictBtn.classList.remove('btn-loading');
      predictBtn.disabled = false;
    }
  });

  // Render Prediction Result
  function displayResult(res) {
    resultSpecies.textContent = res.species;
    resultScientific.textContent = res.scientific_name;
    resultIcon.textContent = res.icon;
    resultDesc.textContent = res.description;

    if (res.gradient) {
      resultBanner.style.background = res.gradient;
    }

    // Update confidence meter
    if (res.confidence !== null && res.confidence !== undefined) {
      confidenceVal.textContent = `${res.confidence}%`;
      confidenceBar.style.width = `${res.confidence}%`;
      confidenceBar.style.background = res.color || 'var(--primary-accent)';
    }

    // Update probability distribution breakdown
    if (probList && res.probabilities) {
      probList.innerHTML = '';
      for (const [speciesName, prob] of Object.entries(res.probabilities)) {
        const item = document.createElement('div');
        item.className = 'prob-item';
        item.innerHTML = `
          <div class="prob-item-header">
            <span class="prob-species-name">${speciesName}</span>
            <span class="prob-species-pct">${prob}%</span>
          </div>
          <div class="prob-track">
            <div class="prob-fill" style="width: ${prob}%;"></div>
          </div>
        `;
        probList.appendChild(item);
      }
    }

    // Transition view
    resultPlaceholder.classList.add('hidden');
    resultContent.classList.remove('hidden');

    // Smooth scroll for mobile viewports
    if (window.innerWidth <= 860) {
      resultContent.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }
});
