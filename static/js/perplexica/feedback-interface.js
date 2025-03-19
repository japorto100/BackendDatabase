/**
 * Document Processing Feedback Interface
 * 
 * This module implements the UI for collecting user feedback on document processing results.
 * The feedback is used to adaptively improve adapter selection and processing quality.
 */

class DocumentFeedbackManager {
    constructor() {
        this.initialized = false;
        this.documentId = null;
        this.adapterName = null;
        this.processingTimestamp = null;
    }

    /**
     * Initialize the feedback interface for a specific document result
     * @param {Object} config - Configuration with document details
     */
    initialize(config) {
        this.documentId = config.documentId;
        this.adapterName = config.adapterName;
        this.processingTimestamp = config.processingTimestamp;
        this.documentType = config.documentType || 'unknown';
        this.apiEndpoint = config.apiEndpoint || '/api/document-feedback/';
        
        this.renderFeedbackForm();
        this.attachEventListeners();
        
        this.initialized = true;
    }

    /**
     * Render the feedback form in the designated container
     */
    renderFeedbackForm() {
        const container = document.getElementById('document-feedback-container');
        if (!container) {
            console.error('Feedback container not found');
            return;
        }

        const formHtml = `
            <div class="feedback-panel">
                <h3>How was the document processing quality?</h3>
                <div class="rating-container">
                    <div class="stars">
                        <span class="star" data-rating="1">★</span>
                        <span class="star" data-rating="2">★</span>
                        <span class="star" data-rating="3">★</span>
                        <span class="star" data-rating="4">★</span>
                        <span class="star" data-rating="5">★</span>
                    </div>
                    <span class="rating-text">Select a rating</span>
                </div>
                
                <div class="quality-checkboxes">
                    <h4>What specific aspects could be improved?</h4>
                    <div class="checkbox-group">
                        <label><input type="checkbox" name="aspect" value="text_accuracy"> Text accuracy</label>
                        <label><input type="checkbox" name="aspect" value="image_quality"> Image quality</label>
                        <label><input type="checkbox" name="aspect" value="table_detection"> Table detection</label>
                        <label><input type="checkbox" name="aspect" value="form_recognition"> Form recognition</label>
                        <label><input type="checkbox" name="aspect" value="document_structure"> Document structure</label>
                    </div>
                </div>
                
                <div class="feedback-comments">
                    <label for="feedback-text">Additional comments:</label>
                    <textarea id="feedback-text" rows="3" placeholder="Please share any additional feedback on the document processing..."></textarea>
                </div>
                
                <div class="feedback-actions">
                    <button id="submit-feedback" class="btn btn-primary">Submit Feedback</button>
                </div>
                
                <div id="feedback-status" class="feedback-status"></div>
            </div>
        `;
        
        container.innerHTML = formHtml;
    }

    /**
     * Attach event listeners to the feedback form elements
     */
    attachEventListeners() {
        // Star rating
        const stars = document.querySelectorAll('.star');
        stars.forEach(star => {
            star.addEventListener('click', (e) => {
                const rating = parseInt(e.target.getAttribute('data-rating'));
                this.setRating(rating);
            });
            
            star.addEventListener('mouseover', (e) => {
                const rating = parseInt(e.target.getAttribute('data-rating'));
                this.highlightStars(rating);
            });
        });
        
        const starsContainer = document.querySelector('.stars');
        if (starsContainer) {
            starsContainer.addEventListener('mouseout', () => {
                this.resetStarHighlight();
            });
        }
        
        // Submit button
        const submitButton = document.getElementById('submit-feedback');
        if (submitButton) {
            submitButton.addEventListener('click', () => {
                this.submitFeedback();
            });
        }
    }

    /**
     * Set the selected rating
     * @param {number} rating - Rating from 1-5 
     */
    setRating(rating) {
        this.rating = rating;
        this.highlightStars(rating, true);
        
        const ratingText = document.querySelector('.rating-text');
        if (ratingText) {
            const ratingDescriptions = [
                'Poor', 'Fair', 'Good', 'Very Good', 'Excellent'
            ];
            ratingText.textContent = ratingDescriptions[rating - 1];
        }
    }

    /**
     * Highlight stars up to the specified rating
     * @param {number} rating - Rating from 1-5
     * @param {boolean} permanent - Whether this is a permanent selection
     */
    highlightStars(rating, permanent = false) {
        const stars = document.querySelectorAll('.star');
        
        stars.forEach((star, index) => {
            if (index < rating) {
                star.classList.add('active');
                if (permanent) {
                    star.classList.add('selected');
                }
            } else {
                star.classList.remove('active');
                if (permanent) {
                    star.classList.remove('selected');
                }
            }
        });
    }

    /**
     * Reset star highlighting to the current rating
     */
    resetStarHighlight() {
        if (this.rating) {
            this.highlightStars(this.rating);
        } else {
            const stars = document.querySelectorAll('.star');
            stars.forEach(star => {
                if (!star.classList.contains('selected')) {
                    star.classList.remove('active');
                }
            });
        }
    }

    /**
     * Submit feedback to the backend API
     */
    submitFeedback() {
        if (!this.rating) {
            this.showStatus('Please select a rating before submitting', 'error');
            return;
        }
        
        // Collect selected aspects that need improvement
        const selectedAspects = [];
        document.querySelectorAll('input[name="aspect"]:checked').forEach(checkbox => {
            selectedAspects.push(checkbox.value);
        });
        
        // Get comments
        const comments = document.getElementById('feedback-text').value;
        
        // Prepare feedback data
        const feedbackData = {
            document_id: this.documentId,
            adapter_name: this.adapterName,
            rating: this.rating,
            processing_timestamp: this.processingTimestamp,
            document_type: this.documentType,
            improvement_aspects: selectedAspects,
            feedback_text: comments
        };
        
        // Submit to API
        this.showStatus('Submitting feedback...', 'info');
        
        fetch(this.apiEndpoint, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'X-CSRFToken': this.getCsrfToken()
            },
            body: JSON.stringify(feedbackData)
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(data => {
            this.showStatus('Thank you for your feedback!', 'success');
            this.disableForm();
            
            // Trigger event for parent application
            const event = new CustomEvent('documentFeedbackSubmitted', { 
                detail: { feedback: feedbackData, response: data } 
            });
            document.dispatchEvent(event);
        })
        .catch(error => {
            console.error('Error submitting feedback:', error);
            this.showStatus('Error submitting feedback. Please try again.', 'error');
        });
    }

    /**
     * Show status message in the feedback form
     * @param {string} message - Status message to display
     * @param {string} type - Message type (info, success, error)
     */
    showStatus(message, type = 'info') {
        const statusElement = document.getElementById('feedback-status');
        if (statusElement) {
            statusElement.textContent = message;
            statusElement.className = 'feedback-status ' + type;
        }
    }

    /**
     * Disable form after successful submission
     */
    disableForm() {
        const stars = document.querySelectorAll('.star');
        stars.forEach(star => {
            star.style.pointerEvents = 'none';
        });
        
        const checkboxes = document.querySelectorAll('input[name="aspect"]');
        checkboxes.forEach(checkbox => {
            checkbox.disabled = true;
        });
        
        const textarea = document.getElementById('feedback-text');
        if (textarea) {
            textarea.disabled = true;
        }
        
        const submitButton = document.getElementById('submit-feedback');
        if (submitButton) {
            submitButton.disabled = true;
        }
    }

    /**
     * Get CSRF token from cookies for secure POST requests
     * @returns {string} CSRF token
     */
    getCsrfToken() {
        const name = 'csrftoken';
        const value = `; ${document.cookie}`;
        const parts = value.split(`; ${name}=`);
        if (parts.length === 2) {
            return parts.pop().split(';').shift();
        }
        return '';
    }
}

// CSS styles for the feedback interface
document.addEventListener('DOMContentLoaded', function() {
    const style = document.createElement('style');
    style.textContent = `
        .feedback-panel {
            background-color: #f8f9fa;
            border-radius: 8px;
            padding: 20px;
            margin: 20px 0;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        
        .rating-container {
            display: flex;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .stars {
            display: flex;
            margin-right: 15px;
        }
        
        .star {
            font-size: 28px;
            color: #ccc;
            cursor: pointer;
            transition: color 0.2s;
            user-select: none;
        }
        
        .star.active, .star.selected {
            color: #FFD700;
        }
        
        .rating-text {
            font-size: 16px;
            color: #555;
        }
        
        .checkbox-group {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 10px;
            margin-bottom: 20px;
        }
        
        .checkbox-group label {
            display: flex;
            align-items: center;
            font-size: 14px;
        }
        
        .checkbox-group input {
            margin-right: 8px;
        }
        
        .feedback-comments {
            margin-bottom: 20px;
        }
        
        .feedback-comments label {
            display: block;
            margin-bottom: 5px;
            font-weight: 500;
        }
        
        .feedback-comments textarea {
            width: 100%;
            padding: 8px;
            border: 1px solid #ccc;
            border-radius: 4px;
        }
        
        .feedback-actions {
            margin-bottom: 15px;
        }
        
        .feedback-status {
            padding: 10px;
            border-radius: 4px;
            font-size: 14px;
        }
        
        .feedback-status.success {
            background-color: #d4edda;
            color: #155724;
        }
        
        .feedback-status.error {
            background-color: #f8d7da;
            color: #721c24;
        }
        
        .feedback-status.info {
            background-color: #d1ecf1;
            color: #0c5460;
        }
    `;
    document.head.appendChild(style);
});

// Create global instance
const documentFeedback = new DocumentFeedbackManager();

// Export for module usage
if (typeof module !== 'undefined' && module.exports) {
    module.exports = documentFeedback;
}
