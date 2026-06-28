/**
 * Main JavaScript file for Spam Email Classifier
 * Handles frontend interactions and API calls
 */

// API Base URL
const API_BASE = '/api';

/**
 * Make API call to classify text
 * @param {string} text - Text to classify
 * @param {string} model - Model type ('naive_bayes' or 'svm')
 * @returns {Promise} API response
 */
function classifyText(text, model = 'naive_bayes') {
    return fetch(`${API_BASE}/classify`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            text: text,
            model: model
        })
    }).then(response => {
        if (!response.ok) {
            throw new Error(`API error: ${response.statusCode}`);
        }
        return response.json();
    });
}

/**
 * Make API call to classify multiple texts
 * @param {array} texts - Array of texts to classify
 * @param {string} model - Model type
 * @returns {Promise} API response
 */
function classifyBatch(texts, model = 'naive_bayes') {
    return fetch(`${API_BASE}/classify-batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({
            texts: texts,
            model: model
        })
    }).then(response => response.json());
}

/**
 * Get model metrics
 * @returns {Promise} API response with metrics
 */
function getMetrics() {
    return fetch(`${API_BASE}/metrics`)
        .then(response => response.json());
}

/**
 * Compare predictions from both models
 * @param {string} text - Text to classify
 * @returns {Promise} API response with both predictions
 */
function compareModels(text) {
    return fetch(`${API_BASE}/compare`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    }).then(response => response.json());
}

/**
 * Preprocess text
 * @param {string} text - Text to preprocess
 * @returns {Promise} API response with preprocessed text
 */
function preprocessText(text) {
    return fetch(`${API_BASE}/preprocess`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify({ text: text })
    }).then(response => response.json());
}

/**
 * Get application statistics
 * @returns {Promise} API response with stats
 */
function getStats() {
    return fetch(`${API_BASE}/stats`)
        .then(response => response.json());
}

/**
 * Format percentage for display
 * @param {number} value - Value between 0 and 1
 * @returns {string} Formatted percentage
 */
function formatPercent(value) {
    return (value * 100).toFixed(1) + '%';
}

/**
 * Display error message
 * @param {string} message - Error message
 */
function showError(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-danger alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('main') || document.body;
    container.insertBefore(alert, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => alert.remove(), 5000);
}

/**
 * Display success message
 * @param {string} message - Success message
 */
function showSuccess(message) {
    const alert = document.createElement('div');
    alert.className = 'alert alert-success alert-dismissible fade show';
    alert.innerHTML = `
        ${message}
        <button type="button" class="btn-close" data-bs-dismiss="alert"></button>
    `;
    
    const container = document.querySelector('main') || document.body;
    container.insertBefore(alert, container.firstChild);
    
    // Auto-dismiss after 5 seconds
    setTimeout(() => alert.remove(), 5000);
}

// Document ready
$(document).ready(function() {
    // Load stats on page load
    loadStats();
});

/**
 * Load and display application statistics
 */
function loadStats() {
    getStats()
        .then(data => {
            console.log('Stats loaded:', data);
        })
        .catch(error => {
            console.error('Error loading stats:', error);
        });
}

// Export functions for use in other scripts
window.SpamClassifier = {
    classifyText,
    classifyBatch,
    getMetrics,
    compareModels,
    preprocessText,
    getStats,
    formatPercent,
    showError,
    showSuccess
};
