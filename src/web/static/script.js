// Customer Churn Prediction App JavaScript

class ChurnPredictionApp {
    constructor() {
        this.apiBaseUrl = window.location.origin;
        this.chart = null;
        this.init();
    }

    async init() {
        try {
            await this.loadMetadata();
            await this.checkModelStatus();
            this.setupEventListeners();
            this.setupFormValidation();
        } catch (error) {
            console.error('Initialization error:', error);
            this.showError('Failed to initialize the application');
        }
    }

    async loadMetadata() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/metadata`);
            if (!response.ok) throw new Error('Failed to load metadata');
            
            const metadata = await response.json();
            this.populateDropdowns(metadata);
        } catch (error) {
            console.error('Error loading metadata:', error);
            this.showError('Failed to load form options');
        }
    }

    async checkModelStatus() {
        try {
            const response = await fetch(`${this.apiBaseUrl}/health`);
            if (!response.ok) throw new Error('Failed to check model status');
            
            const health = await response.json();
            this.updateModelStatus(health);
        } catch (error) {
            console.error('Error checking model status:', error);
            document.getElementById('model-status').textContent = 'Error';
        }
    }

    populateDropdowns(metadata) {
        // Populate states
        const stateSelect = document.getElementById('state');
        metadata.states.forEach(state => {
            const option = document.createElement('option');
            option.value = state;
            option.textContent = state;
            stateSelect.appendChild(option);
        });

        // Populate value deals
        const valueDealsSelect = document.getElementById('value_deal');
        metadata.value_deals.forEach(deal => {
            const option = document.createElement('option');
            option.value = deal;
            option.textContent = deal;
            valueDealsSelect.appendChild(option);
        });

        // Populate contracts
        const contractSelect = document.getElementById('contract');
        metadata.contracts.forEach(contract => {
            const option = document.createElement('option');
            option.value = contract;
            option.textContent = contract;
            contractSelect.appendChild(option);
        });

        // Populate payment methods
        const paymentMethodSelect = document.getElementById('payment_method');
        metadata.payment_methods.forEach(method => {
            const option = document.createElement('option');
            option.value = method;
            option.textContent = method;
            paymentMethodSelect.appendChild(option);
        });

        // Populate internet types
        const internetTypeSelect = document.getElementById('internet_type');
        metadata.internet_types.forEach(type => {
            const option = document.createElement('option');
            option.value = type;
            option.textContent = type;
            internetTypeSelect.appendChild(option);
        });
    }

    updateModelStatus(health) {
        const statusElement = document.getElementById('model-status');
        const featuresElement = document.getElementById('features-count');

        if (health.model_loaded) {
            statusElement.textContent = 'Ready';
            statusElement.style.color = '#28a745';
            featuresElement.textContent = health.features_count || '-';
        } else {
            statusElement.textContent = 'Not Available';
            statusElement.style.color = '#dc3545';
        }
    }

    setupEventListeners() {
        // Form submission
        document.getElementById('churn-form').addEventListener('submit', (e) => {
            e.preventDefault();
            this.handlePrediction();
        });

        // Modal close events
        document.getElementById('close-error-modal').addEventListener('click', () => {
            this.hideError();
        });
        document.getElementById('close-error-btn').addEventListener('click', () => {
            this.hideError();
        });

        // Auto-calculate total revenue when financial fields change
        const financialFields = ['total_charges', 'total_refunds'];
        financialFields.forEach(fieldId => {
            document.getElementById(fieldId).addEventListener('input', () => {
                this.calculateTotalRevenue();
            });
        });

        // Auto-calculate total charges based on tenure and monthly charge
        const chargeFields = ['monthly_charge', 'tenure_in_months'];
        chargeFields.forEach(fieldId => {
            document.getElementById(fieldId).addEventListener('input', () => {
                this.calculateTotalCharges();
            });
        });
    }

    setupFormValidation() {
        const form = document.getElementById('churn-form');
        const inputs = form.querySelectorAll('input, select');

        inputs.forEach(input => {
            input.addEventListener('blur', () => {
                this.validateField(input);
            });
            
            input.addEventListener('input', () => {
                this.clearFieldError(input);
            });
        });
    }

    validateField(field) {
        const value = field.value.trim();
        const fieldName = field.name;
        let isValid = true;
        let errorMessage = '';

        // Required field validation
        if (field.hasAttribute('required') && !value) {
            isValid = false;
            errorMessage = 'This field is required';
        }

        // Specific field validations
        if (isValid && value) {
            switch (fieldName) {
                case 'age':
                    if (value < 18 || value > 100) {
                        isValid = false;
                        errorMessage = 'Age must be between 18 and 100';
                    }
                    break;
                case 'monthly_charge':
                    if (value < 0 || value > 1000) {
                        isValid = false;
                        errorMessage = 'Monthly charge must be between $0 and $1000';
                    }
                    break;
                case 'total_charges':
                    const monthlyCharge = parseFloat(document.getElementById('monthly_charge').value) || 0;
                    const tenure = parseInt(document.getElementById('tenure_in_months').value) || 0;
                    const expectedMin = monthlyCharge * tenure * 0.5;
                    if (value < expectedMin) {
                        isValid = false;
                        errorMessage = 'Total charges seem too low for the given monthly charge and tenure';
                    }
                    break;
            }
        }

        if (!isValid) {
            this.showFieldError(field, errorMessage);
        } else {
            this.clearFieldError(field);
        }

        return isValid;
    }

    showFieldError(field, message) {
        field.classList.add('error');
        
        // Remove existing error message
        const existingError = field.parentNode.querySelector('.error-message');
        if (existingError) {
            existingError.remove();
        }

        // Add new error message
        const errorElement = document.createElement('div');
        errorElement.className = 'error-message';
        errorElement.textContent = message;
        field.parentNode.appendChild(errorElement);
    }

    clearFieldError(field) {
        field.classList.remove('error');
        const errorMessage = field.parentNode.querySelector('.error-message');
        if (errorMessage) {
            errorMessage.remove();
        }
    }

    calculateTotalCharges() {
        const monthlyCharge = parseFloat(document.getElementById('monthly_charge').value) || 0;
        const tenure = parseInt(document.getElementById('tenure_in_months').value) || 0;
        
        if (monthlyCharge > 0 && tenure > 0) {
            const estimatedTotal = monthlyCharge * tenure;
            const totalChargesField = document.getElementById('total_charges');
            
            // Only auto-fill if the field is empty
            if (!totalChargesField.value) {
                totalChargesField.value = estimatedTotal.toFixed(2);
                this.calculateTotalRevenue();
            }
        }
    }

    calculateTotalRevenue() {
        const totalCharges = parseFloat(document.getElementById('total_charges').value) || 0;
        const totalRefunds = parseFloat(document.getElementById('total_refunds').value) || 0;
        const extraDataCharges = parseFloat(document.getElementById('total_extra_data_charges').value) || 0;
        const longDistanceCharges = parseFloat(document.getElementById('total_long_distance_charges').value) || 0;
        
        const totalRevenue = totalCharges - totalRefunds + extraDataCharges + longDistanceCharges;
        document.getElementById('total_revenue').value = Math.max(0, totalRevenue).toFixed(2);
    }

    async handlePrediction() {
        // Validate form
        const form = document.getElementById('churn-form');
        const inputs = form.querySelectorAll('input, select');
        let isFormValid = true;

        inputs.forEach(input => {
            if (!this.validateField(input)) {
                isFormValid = false;
            }
        });

        if (!isFormValid) {
            this.showError('Please fix the validation errors before submitting');
            return;
        }

        // Collect form data
        const formData = new FormData(form);
        const data = {};
        
        for (let [key, value] of formData.entries()) {
            // Convert numeric fields
            if (['age', 'number_of_referrals', 'tenure_in_months'].includes(key)) {
                data[key] = parseInt(value);
            } else if ([
                'monthly_charge', 'total_charges', 'total_refunds', 
                'total_extra_data_charges', 'total_long_distance_charges', 'total_revenue'
            ].includes(key)) {
                data[key] = parseFloat(value);
            } else {
                data[key] = value;
            }
        }

        try {
            this.showLoading();
            
            const response = await fetch(`${this.apiBaseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(data)
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.detail || 'Prediction failed');
            }

            const result = await response.json();
            this.displayResults(result);
            
        } catch (error) {
            console.error('Prediction error:', error);
            this.showError(error.message || 'Failed to make prediction');
        } finally {
            this.hideLoading();
        }
    }

    displayResults(result) {
        // Show results section
        const resultsSection = document.getElementById('results-section');
        resultsSection.style.display = 'block';
        resultsSection.scrollIntoView({ behavior: 'smooth' });

        // Update prediction result
        const predictionCard = document.getElementById('prediction-card');
        const predictionResult = document.getElementById('prediction-result');
        const predictionIcon = predictionCard.querySelector('.result-icon i');

        predictionResult.textContent = result.prediction_label;
        
        if (result.prediction === 1) {
            predictionCard.className = 'result-card danger';
            predictionIcon.className = 'fas fa-exclamation-triangle';
            predictionIcon.parentElement.className = 'result-icon danger';
        } else {
            predictionCard.className = 'result-card success';
            predictionIcon.className = 'fas fa-check-circle';
            predictionIcon.parentElement.className = 'result-icon success';
        }

        // Update probability result
        const probabilityCard = document.getElementById('probability-card');
        const probabilityResult = document.getElementById('probability-result');
        const probabilityIcon = probabilityCard.querySelector('.result-icon i');

        probabilityResult.textContent = (result.probability * 100).toFixed(1) + '%';
        
        if (result.probability >= 0.7) {
            probabilityCard.className = 'result-card danger';
            probabilityIcon.parentElement.className = 'result-icon danger';
        } else if (result.probability >= 0.3) {
            probabilityCard.className = 'result-card warning';
            probabilityIcon.parentElement.className = 'result-icon warning';
        } else {
            probabilityCard.className = 'result-card success';
            probabilityIcon.parentElement.className = 'result-icon success';
        }

        // Update confidence result
        const confidenceCard = document.getElementById('confidence-card');
        const confidenceResult = document.getElementById('confidence-result');
        const confidenceIcon = confidenceCard.querySelector('.result-icon i');

        confidenceResult.textContent = result.confidence;
        
        switch (result.confidence) {
            case 'High':
                confidenceCard.className = 'result-card success';
                confidenceIcon.parentElement.className = 'result-icon success';
                break;
            case 'Medium':
                confidenceCard.className = 'result-card warning';
                confidenceIcon.parentElement.className = 'result-icon warning';
                break;
            case 'Low':
                confidenceCard.className = 'result-card danger';
                confidenceIcon.parentElement.className = 'result-icon danger';
                break;
        }

        // Display feature importance chart
        if (result.feature_importance && result.feature_importance.length > 0) {
            this.displayFeatureImportance(result.feature_importance);
        }

        // Display recommendations
        this.displayRecommendations(result);

        // Add animation
        resultsSection.classList.add('fade-in');
    }

    displayFeatureImportance(featureImportance) {
        const container = document.getElementById('feature-importance-container');
        container.style.display = 'block';

        const ctx = document.getElementById('feature-chart').getContext('2d');
        
        // Destroy existing chart if it exists
        if (this.chart) {
            this.chart.destroy();
        }

        const labels = featureImportance.slice(0, 8).map(item => 
            item.feature.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
        );
        const data = featureImportance.slice(0, 8).map(item => item.importance);

        this.chart = new Chart(ctx, {
            type: 'horizontalBar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Feature Importance',
                    data: data,
                    backgroundColor: [
                        'rgba(46, 134, 171, 0.8)',
                        'rgba(162, 59, 114, 0.8)',
                        'rgba(241, 143, 1, 0.8)',
                        'rgba(40, 167, 69, 0.8)',
                        'rgba(255, 193, 7, 0.8)',
                        'rgba(220, 53, 69, 0.8)',
                        'rgba(108, 117, 125, 0.8)',
                        'rgba(102, 126, 234, 0.8)'
                    ],
                    borderColor: [
                        'rgba(46, 134, 171, 1)',
                        'rgba(162, 59, 114, 1)',
                        'rgba(241, 143, 1, 1)',
                        'rgba(40, 167, 69, 1)',
                        'rgba(255, 193, 7, 1)',
                        'rgba(220, 53, 69, 1)',
                        'rgba(108, 117, 125, 1)',
                        'rgba(102, 126, 234, 1)'
                    ],
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    xAxes: [{
                        beginAtZero: true,
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(1) + '%';
                            }
                        }
                    }]
                },
                legend: {
                    display: false
                },
                tooltips: {
                    callbacks: {
                        label: function(tooltipItem) {
                            return 'Importance: ' + (tooltipItem.xLabel * 100).toFixed(2) + '%';
                        }
                    }
                }
            }
        });
    }

    displayRecommendations(result) {
        const container = document.getElementById('recommendations-list');
        container.innerHTML = '';

        let recommendations = [];
        
        if (result.prediction === 1) {
            // High churn risk recommendations
            recommendations = [
                { text: 'Immediate Action Required: Contact customer within 24 hours', priority: 'high-priority' },
                { text: 'Retention Offer: Consider special pricing or value deals', priority: 'high-priority' },
                { text: 'Personal Touch: Assign dedicated account manager', priority: 'medium-priority' },
                { text: 'Service Review: Analyze and address service issues', priority: 'medium-priority' },
                { text: 'Usage Analysis: Review and optimize service plan', priority: 'low-priority' }
            ];
        } else {
            // Low churn risk recommendations
            recommendations = [
                { text: 'Upselling Opportunity: Customer is satisfied, consider premium services', priority: 'low-priority' },
                { text: 'Growth Potential: Explore additional service offerings', priority: 'low-priority' },
                { text: 'Loyalty Program: Enroll in rewards program', priority: 'low-priority' },
                { text: 'Feedback Collection: Gather insights for service improvement', priority: 'medium-priority' },
                { text: 'Regular Check-ins: Maintain positive relationship', priority: 'low-priority' }
            ];
        }

        recommendations.forEach(rec => {
            const recElement = document.createElement('div');
            recElement.className = `recommendation-item ${rec.priority}`;
            recElement.textContent = rec.text;
            container.appendChild(recElement);
        });
    }

    showLoading() {
        document.getElementById('loading-overlay').style.display = 'flex';
        document.getElementById('predict-btn').disabled = true;
    }

    hideLoading() {
        document.getElementById('loading-overlay').style.display = 'none';
        document.getElementById('predict-btn').disabled = false;
    }

    showError(message) {
        document.getElementById('error-message').textContent = message;
        document.getElementById('error-modal').style.display = 'flex';
    }

    hideError() {
        document.getElementById('error-modal').style.display = 'none';
    }
}

// Initialize the app when the DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new ChurnPredictionApp();
});
