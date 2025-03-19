document.addEventListener('DOMContentLoaded', function() {
    // Find the canton selector
    const cantonSelect = document.getElementById('id_canton');
    const kwhrateInput = document.getElementById('id_kwh_rate');
    const updateButton = document.getElementById('update-from-canton');
    
    if (updateButton && cantonSelect && kwhrateInput) {
        updateButton.addEventListener('click', function() {
            const selectedCanton = cantonSelect.value;
            if (!selectedCanton) {
                alert('Please select a canton first');
                return;
            }
            
            // Get the default rate for the selected canton
            fetch(`/admin/models_app/electricitysettings/get-canton-rate/${selectedCanton}/`)
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        kwhrateInput.value = data.rate;
                    } else {
                        alert('Error: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Error fetching canton rate:', error);
                    alert('Failed to get rate for selected canton');
                });
        });
    }
}); 