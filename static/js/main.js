// Update range input values
document.addEventListener('DOMContentLoaded', function() {
    const ageSlider = document.getElementById('age-slider');
    const incomeSlider = document.getElementById('income-slider');
    const spendingSlider = document.getElementById('spending-slider');
    
    const ageInput = document.getElementById('age');
    const incomeInput = document.getElementById('income');
    const spendingInput = document.getElementById('spending');
    
    // Sync sliders with inputs
    ageSlider.addEventListener('input', function() {
        ageInput.value = this.value;
    });
    
    incomeSlider.addEventListener('input', function() {
        incomeInput.value = this.value;
    });
    
    spendingSlider.addEventListener('input', function() {
        spendingInput.value = this.value;
    });
    
    // Sync inputs with sliders
    ageInput.addEventListener('input', function() {
        ageSlider.value = this.value;
    });
    
    incomeInput.addEventListener('input', function() {
        incomeSlider.value = this.value;
    });
    
    spendingInput.addEventListener('input', function() {
        spendingSlider.value = this.value;
    });
    
    // Form submission
    const segmentationForm = document.getElementById('segmentation-form');
    segmentationForm.addEventListener('submit', function(e) {
        e.preventDefault();
        
        const formData = new FormData(segmentationForm);
        
        fetch('/predict', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                alert('Error: ' + data.error);
                return;
            }
            
            // Display results
            document.getElementById('cluster-number').textContent = data.cluster;
            document.getElementById('cluster-name').textContent = data.cluster_name;
            document.getElementById('cluster-description').textContent = data.cluster_description;
            document.getElementById('result-age').textContent = data.age;
            document.getElementById('result-income').textContent = data.income;
            document.getElementById('result-spending').textContent = data.spending;
            document.getElementById('scatterplot').src = 'data:image/png;base64,' + data.plot_url;
            
            // Populate cluster centers table
            const tableBody = document.getElementById('cluster-centers-table');
            tableBody.innerHTML = '';
            
            data.centers.forEach((center, index) => {
                const row = document.createElement('tr');
                row.innerHTML = `
                    <td>${index}</td>
                    <td>${center[0].toFixed(1)}</td>
                    <td>$${center[1].toFixed(1)}k</td>
                    <td>${center[2].toFixed(1)}</td>
                `;
                tableBody.appendChild(row);
            });
            
            // Show results section
            document.getElementById('results').style.display = 'block';
            
            // Scroll to results
            document.getElementById('results').scrollIntoView({ behavior: 'smooth' });
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while processing your request. Please try again.');
        });
    });
    
    // Smooth scrolling for navigation links
    document.querySelectorAll('a[href^="#"]').forEach(anchor => {
        anchor.addEventListener('click', function (e) {
            e.preventDefault();
            const target = document.querySelector(this.getAttribute('href'));
            if (target) {
                target.scrollIntoView({
                    behavior: 'smooth'
                });
            }
        });
    });
    
    // Back to top button
    document.querySelector('.back-to-top').addEventListener('click', function(e) {
        e.preventDefault();
        window.scrollTo({
            top: 0,
            behavior: 'smooth'
        });
    });
});
