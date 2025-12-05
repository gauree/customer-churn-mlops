document.addEventListener('DOMContentLoaded', () => {
  const form = document.getElementById('churn-form');
  const resultSection = document.getElementById('result-section');
  const predictionText = document.getElementById('prediction-text');
  const probabilityText = document.getElementById('probability-text');
  const errorSection = document.getElementById('error-section');
  const errorText = document.getElementById('error-text');
  const submitBtn = document.getElementById('submit-btn');

  function showError(message) {
    errorText.textContent = message;
    errorSection.classList.remove('hidden');
    resultSection.classList.add('hidden');
  }

  function showResult(prediction, probability) {
    const isChurn = prediction === 1;
    predictionText.textContent = isChurn
      ? 'Customer is likely to churn.'
      : 'Customer is unlikely to churn.';

    if (typeof probability === 'number' && !Number.isNaN(probability)) {
      const pct = (probability * 100).toFixed(1);
      probabilityText.textContent = `Estimated churn probability: ${pct}%`;
    } else {
      probabilityText.textContent = '';
    }

    errorSection.classList.add('hidden');
    resultSection.classList.remove('hidden');
  }

  form.addEventListener('submit', async (event) => {
    event.preventDefault();

    submitBtn.disabled = true;
    submitBtn.textContent = 'Predicting...';

    try {
      const formData = new FormData(form);
      const data = {
        gender: formData.get('gender'),
        SeniorCitizen: Number(formData.get('SeniorCitizen')),
        Partner: formData.get('Partner'),
        Dependents: formData.get('Dependents'),
        tenure: Number(formData.get('tenure')),
        PhoneService: formData.get('PhoneService'),
        MultipleLines: formData.get('MultipleLines'),
        InternetService: formData.get('InternetService'),
        OnlineSecurity: formData.get('OnlineSecurity'),
        OnlineBackup: formData.get('OnlineBackup'),
        DeviceProtection: formData.get('DeviceProtection'),
        TechSupport: formData.get('TechSupport'),
        StreamingTV: formData.get('StreamingTV'),
        StreamingMovies: formData.get('StreamingMovies'),
        Contract: formData.get('Contract'),
        PaperlessBilling: formData.get('PaperlessBilling'),
        PaymentMethod: formData.get('PaymentMethod'),
        MonthlyCharges: parseFloat(formData.get('MonthlyCharges')),
        TotalCharges: parseFloat(formData.get('TotalCharges')),
      };

      const payload = { data: [data] };

      const response = await fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      let json;
      try {
        json = await response.json();
      } catch {
        json = null;
      }

      if (!response.ok) {
        if (json && json.error) {
          showError(json.error);
        } else {
          showError(`Request failed with status ${response.status}`);
        }
        return;
      }

      const prediction = Array.isArray(json.predictions)
        ? json.predictions[0]
        : null;
      const probability = Array.isArray(json.churn_probability)
        ? json.churn_probability[0]
        : null;

      if (prediction === null || prediction === undefined) {
        showError('No prediction value returned from API.');
        return;
      }

      showResult(prediction, probability);
    } catch (err) {
      showError(`Unexpected error: ${err}`);
    } finally {
      submitBtn.disabled = false;
      submitBtn.textContent = 'Predict Churn';
    }
  });
});
