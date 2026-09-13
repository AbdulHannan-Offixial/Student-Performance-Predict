// Update this once the backend is deployed. During local development this
// matches the port exposed by docker-compose.yml (see the backend folder).
const API_BASE_URL = 'http://localhost:8000';

// These are the only fields that should be sent as numbers — everything
// else in the form is a category (select) and stays a string.
const NUMERIC_FIELDS = [
  'Hours_Studied',
  'Attendance',
  'Previous_Scores',
  'Tutoring_Sessions',
  'Sleep_Hours',
  'Physical_Activity',
];

const form = document.getElementById('predictorForm');
const submitButton = document.getElementById('submitButton');

const resultPlaceholder = document.getElementById('resultPlaceholder');
const resultContent = document.getElementById('resultContent');
const resultError = document.getElementById('resultError');
const scoreValue = document.getElementById('scoreValue');
const scoreBarFill = document.getElementById('scoreBarFill');

function buildPayload(formData) {
  const payload = {};
  for (const [key, value] of formData.entries()) {
    payload[key] = NUMERIC_FIELDS.includes(key) ? Number(value) : value;
  }
  return payload;
}

function showLoading() {
  submitButton.disabled = true;
  submitButton.textContent = 'Predicting…';
  resultError.hidden = true;
}

function resetButton() {
  submitButton.disabled = false;
  submitButton.textContent = 'Predict exam score';
}

function showResult(score) {
  resultPlaceholder.hidden = true;
  resultError.hidden = true;
  resultContent.hidden = false;

  scoreValue.textContent = score.toFixed(1);
  scoreBarFill.style.width = `${Math.min(100, Math.max(0, score))}%`;
}

function showError(message) {
  resultPlaceholder.hidden = true;
  resultContent.hidden = true;
  resultError.hidden = false;
  resultError.textContent = message;
}

function extractErrorDetail(errorBody, status) {
  if (!errorBody || !errorBody.detail) {
    return `Something went wrong (status ${status}). Please try again.`;
  }
  // FastAPI validation errors return detail as a list of objects rather
  // than a plain string — handle both shapes.
  if (typeof errorBody.detail === 'string') {
    return errorBody.detail;
  }
  return 'One or more fields were invalid. Please check your entries and try again.';
}

form.addEventListener('submit', async (event) => {
  event.preventDefault();

  if (!form.checkValidity()) {
    form.reportValidity();
    return;
  }

  showLoading();
  const payload = buildPayload(new FormData(form));

  try {
    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const errorBody = await response.json().catch(() => null);
      showError(extractErrorDetail(errorBody, response.status));
      return;
    }

    const data = await response.json();
    showResult(data.predicted_exam_score);
  } catch (networkError) {
    showError(
      `Couldn't reach the prediction server at ${API_BASE_URL}. Make sure the backend is running, then try again.`
    );
  } finally {
    resetButton();
  }
});

// Highlight whichever nav link matches the section currently in view.
// The rootMargin shrinks the detection area to a thin band in the upper
// half of the viewport, so the active link changes right as a section's
// content reaches that point while scrolling — not the instant its edge
// appears at the very bottom of the screen.
const navLinks = document.querySelectorAll('.nav-link');
const sections = document.querySelectorAll('section[id]');

const sectionObserver = new IntersectionObserver(
  (entries) => {
    entries.forEach((entry) => {
      if (!entry.isIntersecting) return;
      const sectionId = entry.target.getAttribute('id');
      navLinks.forEach((link) => {
        link.classList.toggle('active', link.dataset.section === sectionId);
      });
    });
  },
  { rootMargin: '-40% 0px -55% 0px' }
);

sections.forEach((section) => sectionObserver.observe(section));
