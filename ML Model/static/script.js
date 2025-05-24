document.addEventListener('DOMContentLoaded', function() {
    // Form elements
    const form = document.getElementById('churn-form');
    const resetBtn = document.getElementById('reset-btn');
    const resultsSection = document.getElementById('results');
    const predictionValue = document.getElementById('prediction-value');
    const probabilityBar = document.getElementById('probability-bar');
    const probabilityValue = document.getElementById('probability-value');
    const featureImportanceSection = document.getElementById('feature-importance-section');
    const featureImportanceChart = document.getElementById('feature-importance-chart');
    
    // Populate dropdown fields with data from API
    fetch('/metadata')
        .then(response => response.json())
        .then(data => {
            // Populate states dropdown
            const stateDropdown = document.getElementById('state');
            data.states.forEach(state => {
                const option = document.createElement('option');
                option.value = state;
                option.textContent = state;
                stateDropdown.appendChild(option);
            });
            
            // Populate value deal dropdown
            const valueDealDropdown = document.getElementById('value_deal');
            data.value_deals.forEach(deal => {
                const option = document.createElement('option');
                option.value = deal;
                option.textContent = deal;
                valueDealDropdown.appendChild(option);
            });
            
            // Populate contract dropdown
            const contractDropdown = document.getElementById('contract');
            data.contracts.forEach(contract => {
                const option = document.createElement('option');
                option.value = contract;
                option.textContent = contract;
                contractDropdown.appendChild(option);
            });
            
            // Populate payment method dropdown
            const paymentMethodDropdown = document.getElementById('payment_method');
            data.payment_methods.forEach(method => {
                const option = document.createElement('option');
                option.value = method;
                option.textContent = method;
                paymentMethodDropdown.appendChild(option);
            });
        })
        .catch(error => {
            console.error('Error fetching metadata:', error);
            alert('Failed to load form data. Please refresh the page.');
        });
    
    // Form submission handler
    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Show loading state
        document.getElementById('predict-btn').textContent = 'Predicting...';
        document.getElementById('predict-btn').disabled = true;
        
        // Collect form data
        const formData = new FormData(form);
        const jsonData = {};
        
        for (const [key, value] of formData.entries()) {
            // Convert numeric values to numbers
            if (!isNaN(value) && value !== '') {
                jsonData[key] = Number(value);
            } else {
                jsonData[key] = value;
            }
        }
        
        // Log the data being sent (for debugging)
        console.log('Sending data:', jsonData);
        
        // Send prediction request
        fetch('/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(jsonData)
        })
        .then(response => {
            // Check if the response is ok (status in the range 200-299)
            if (!response.ok) {
                // Try to parse the error response as JSON
                return response.json().then(errorData => {
                    // Throw an error with the detailed message from the server
                    throw new Error(errorData.detail || 'Prediction request failed');
                }).catch(jsonError => {
                    // If JSON parsing fails, throw generic error with status
                    throw new Error(`Server error (Status ${response.status}): Please check server logs`);
                });
            }
            return response.json();
        })
        .then(data => {
            // Display prediction result
            resultsSection.classList.remove('hidden');
            
            // Update prediction label
            predictionValue.textContent = data.prediction_label;
            predictionValue.className = data.prediction === 1 ? 'prediction-value churn' : 'prediction-value no-churn';
            
            // Update probability bar
            const probabilityPercentage = (data.probability * 100).toFixed(1);
            probabilityBar.style.width = `${probabilityPercentage}%`;
            probabilityValue.textContent = `${probabilityPercentage}%`;
            
            // Display feature importance if available
            if (data.feature_importance && data.feature_importance.length > 0) {
                featureImportanceSection.classList.remove('hidden');
                
                // Clear previous chart
                featureImportanceChart.innerHTML = '';
                
                // Find max importance for scaling
                const maxImportance = Math.max(...data.feature_importance.map(item => item.importance));
                
                // Create bars for top features
                data.feature_importance.forEach(feature => {
                    // Create bar container
                    const barContainer = document.createElement('div');
                    barContainer.className = 'feature-bar';
                    
                    // Create bar fill
                    const barFill = document.createElement('div');
                    barFill.className = 'feature-bar-fill';
                    const widthPercentage = (feature.importance / maxImportance * 100).toFixed(1);
                    barFill.style.width = '0%'; // Start at 0 for animation
                    
                    // Clean feature name for display
                    let cleanFeatureName = feature.feature
                        .replace(/_/g, ' ')
                        .replace(/value_deal_/g, '')
                        .replace(/contract_/g, '')
                        .replace(/payment_method_/g, '');
                    
                    // Create feature name element
                    const nameElement = document.createElement('div');
                    nameElement.className = 'feature-name';
                    nameElement.textContent = cleanFeatureName;
                    barFill.appendChild(nameElement);
                    
                    // Add bar fill to container
                    barContainer.appendChild(barFill);
                    
                    // Create importance value element
                    const valueElement = document.createElement('div');
                    valueElement.className = 'feature-value';
                    valueElement.textContent = (feature.importance * 100).toFixed(1) + '%';
                    barContainer.appendChild(valueElement);
                    
                    // Add bar to chart
                    featureImportanceChart.appendChild(barContainer);
                    
                    // Animate bar fill after a small delay
                    setTimeout(() => {
                        barFill.style.width = `${widthPercentage}%`;
                    }, 100);
                });
            } else {
                featureImportanceSection.classList.add('hidden');
            }
            
            // Scroll to results
            resultsSection.scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('Error:', error);
            
            // Display a more detailed error message
            alert('Error: ' + error.message);
            
            // Create an error message in the results section
            resultsSection.classList.remove('hidden');
            predictionValue.textContent = 'Error';
            predictionValue.className = 'prediction-value churn';
            probabilityBar.style.width = '0%';
            probabilityValue.textContent = '0%';
            featureImportanceSection.classList.add('hidden');
        })
        .finally(() => {
            // Reset button state
            document.getElementById('predict-btn').textContent = 'Predict Churn';
            document.getElementById('predict-btn').disabled = false;
        });
    });
    
    // Reset form handler
    resetBtn.addEventListener('click', function() {
        form.reset();
        resultsSection.classList.add('hidden');
    });
    
    // Add quick fill button for testing (optional)
    // Uncomment this section if you want to add a quick fill button for testing
    
    const quickFillBtn = document.createElement('button');
    quickFillBtn.type = 'button';
    quickFillBtn.textContent = 'Fill with Sample Data';
    quickFillBtn.className = 'quick-fill-btn';
    quickFillBtn.addEventListener('click', function() {
        // Sample data for quick testing
        const sampleData = {
            gender: 'Male',
            age: 35,
            state: 'Karnataka',
            number_of_referrals: 2,
            tenure_in_months: 52,
            phone_service: 'Yes',
            multiple_lines: 'No',
            internet_service: 'Yes',
            internet_type: 'Fiber Optic',
            online_security: 'No',
            online_backup: 'Yes',
            device_protection_plan: 'No',
            premium_support: 'Yes',
            streaming_tv: 'Yes',
            streaming_movies: 'No',
            streaming_music: 'Yes',
            unlimited_data: 'Yes',
            paperless_billing: 'Yes',
            monthly_charge: 79.95,
            total_charges: 4157.4,
            total_refunds: 0,
            total_extra_data_charges: 0,
            total_long_distance_charges: 107.87,
            total_revenue: 4265.27,
            value_deal: 'No Deal',
            contract: 'Month-to-Month',
            payment_method: 'Credit Card'
        };
        
        // Fill the form with sample data
        Object.keys(sampleData).forEach(key => {
            const element = document.getElementById(key);
            if (element) {
                element.value = sampleData[key];
            }
        });
    });
    
    // Add the button before the form buttons
    document.querySelector('.form-buttons').prepend(quickFillBtn);
    
});