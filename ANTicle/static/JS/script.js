window.onload = function(){
    // Get 'thema' field HTML
    //const themaHtml = `{% for field in thema_html %}{{ field }}{% endfor %}`;
    // Get the generate button
    var generateBtn = document.getElementById('generate');
    // Insert the 'thema' field HTML before the generate button
    generateBtn.insertAdjacentHTML('beforebegin', themaHtml);
};

document.addEventListener('DOMContentLoaded', (event) => {
    const formGroups = Array.from(document.querySelectorAll('.form-group'));
    formGroups.forEach((formGroup, i) => {
        const label = formGroup.querySelector('label');
        const fieldContent = formGroup.querySelector('.field-content');
        label.style.cursor = 'pointer';
        label.addEventListener('click', () => {
            const isDisplayed = fieldContent.style.display !== 'none';
            fieldContent.style.display = isDisplayed ? 'none' : 'block';
        });
    });
});
function changeTab(tab) {
    const inputMenu = document.getElementById('inputMenu');
    const outputMenu = document.getElementById('outputMenu');
    const input = document.getElementById('input');
    const output = document.getElementById('output');

    if (!inputMenu || !outputMenu || !input || !output) {
        console.error("One or more elements could not be found.");
        return;
    }

    switch (tab) {
        case 'input':
            input.classList.remove('hidden');
            output.classList.add('hidden');
            inputMenu.style.backgroundColor = '#41efb4'; // Selected button color
            outputMenu.style.backgroundColor = '#292929'; // Non-selected button color
            break;
        case 'output':
            output.classList.remove('hidden');
            input.classList.add('hidden');
            outputMenu.style.backgroundColor = '#41efb4'; // Selected button color
            inputMenu.style.backgroundColor = '#292929'; // Non-selected button color
            break;
        default:
            console.error(`Unexpected tab value: ${tab}`);
            break;
    }
}


$(document).ready(function() {
    //Spinner
    $("#inputMenu").click();
    $("#inputForm").on('submit', function(e) {
        e.preventDefault();
        $('.spinner-background').css('display', 'block');
            const data = new FormData(this);
            fetch('/my-view/', { method: 'POST', body: data })
                .then(response => response.json())
                .then(function(response) {
                    // Hide spinner
                    $('.spinner-background').css('display', 'none');
                    populateOutputFields(response);
                    document.getElementById('outputMenu').click();
                    document.querySelectorAll('.form-textarea').forEach(autoResize);
                })
                .catch(error => {
                    // Hide spinner
                    $('.spinner-background').css('display', 'none');
                    console.error('Error:', error)
                });
                // Add click event listeners to like and dislike buttons
        $(document).on('click', '.like-btn', function() {
            // Like button logic
            $(this).data('liked', true);  // Save "like" state

            let currElement = this;
            // Extract the text if a h4 was found
            let headline = $(this).parentsUntil('.key-value-container').siblings('.text-lg').text()
            let reaction = "Like";

            // Get the related textarea value.
            let text = $(this).siblings('.form-textarea').val()

            // Print statements for testing
            console.log("Text: ", text);
            console.log("Reaction: ", reaction);
            console.log("Headline: ", headline);

            sendToServer(text, reaction, headline);
        });
        $(document).on('click', '.dislike-btn', function(event) {
            // Dislike button logic
            $(this).data('disliked', true);  // Save "dislike" state

            let currElement = this;
            // Extract the text if a h4 was found
            let headline = $(this).parentsUntil('.key-value-container').siblings('.text-lg').text()
            let reaction = "Dislike";

            // Get the related textarea value.
            let text = $(this).siblings('.form-textarea').val()

            // Print statements for testing
            console.log("Text: ", text);
            console.log("Reaction: ", reaction);
            console.log("Headline: ", headline);

            sendToServer(text, reaction, headline);
        });
    });
    //reset button
    $('#regenerate').click(function() {

    });
    //ToDo!!!!
    $('#regenerate').click(function() {

    });
});

// Function to resize textarea based on content
document.addEventListener('DOMContentLoaded', (event) => {
    // Assuming you have data at this point, otherwise you'll need to call populateOutputFields
    // with your data after you've fetched or generated it.
    // populateOutputFields(retrievedData);
});

//Textboxen
function autoResize(textarea) {
    // Reset the height to ensure the scrollHeight calculation is correct
    textarea.style.height = 'auto';

    // Set a base minimum height for textareas
    const baseMinHeight = 100;

    // For 'Text' and 'Zusatz', set a larger minimum height due to potentially larger content
    const largeMinHeight = 300; // Adjust this value based on your needs
    const isLargeContentKey = textarea.id.includes('Text') || textarea.id.includes('Zusatz');
    const minHeight = isLargeContentKey ? largeMinHeight : baseMinHeight;

    // Adjust the height based on the scrollHeight or the minHeight, whichever is larger
    textarea.style.height = `${Math.max(textarea.scrollHeight, minHeight)}px`;
}



//Output management
function populateOutputFields(data) {
    const keys = Object.keys(data); // Dynamically get the keys from the data
    const resultContainer = document.getElementById('result');
    resultContainer.innerHTML = ''; // Clear existing content

    keys.forEach(function(key, keyIndex) {
        const subKeys = Object.keys(data[key]);
        const keyValueContainer = document.createElement('div');
        keyValueContainer.className = "key-value-container flex items-start py-2"; // Use flexbox for horizontal layout

        const headline = document.createElement('h4');
        headline.textContent = key;
        headline.className = "text-lg font-bold pt-2 cursor-pointer";
        headline.style.minWidth = '150px'; // Ensure consistent width for headlines

        keyValueContainer.appendChild(headline);

        const contentContainer = document.createElement('div');
        contentContainer.style.display = keyIndex === 0 ? 'flex' : 'none'; // Only the first key's content is visible by default
        contentContainer.style.flexDirection = 'column'; // Stack textareas vertically within their container
        contentContainer.style.flexGrow = '1'; // Allow the container to fill available space
        contentContainer.style.paddingLeft = '20px'; // Add some space between the headline and textareas

        if (key === 'Text' || key === 'Zusatz') {
            const allSubKeysValue = Object.values(data[key]).join('\n\n'); // Correctly join the values
            const textareaInfo = createTextarea(key, allSubKeysValue);
            contentContainer.appendChild(textareaInfo.container);
            autoResize(textareaInfo.textarea); // Make sure this is called after the textarea is appended and has content
        } else {
            subKeys.forEach(function(subKey) {
                const textareaInfo = createTextarea(key + '-' + subKey, data[key][subKey]);
                contentContainer.appendChild(textareaInfo.container);
                autoResize(textareaInfo.textarea); // And also here for each subKey's textarea
            });
        }


        keyValueContainer.appendChild(contentContainer);

        headline.addEventListener('click', function() {
            // Toggle the display between 'flex' and 'none'
            contentContainer.style.display = contentContainer.style.display === 'none' ? 'flex' : 'none';
        });

        resultContainer.appendChild(keyValueContainer);
    });
}

function createTextarea(id, value) {
    const container = document.createElement('div');
    container.style.position = 'relative';
    container.style.marginBottom = '10px'; // Space between textareas

    const textarea = document.createElement('textarea');
    textarea.value = value;
    textarea.className = "form-textarea mt-1 block w-full rounded-md shadow-sm";
    textarea.style.resize = 'vertical';
    textarea.style.padding = '2rem 2rem 2rem 0.5rem'; // Padding to accommodate the copy button
    textarea.style.backgroundColor = '#4e4e4e';
    textarea.style.color = '#fbfeff';
    textarea.setAttribute('id', 'textarea-' + id);
    textarea.addEventListener('input', function() { autoResize(textarea); });

    const copyButton = document.createElement('button');
    copyButton.innerHTML = '&#128203;'; // Use a suitable copy icon here
    copyButton.className = "absolute bottom-2 right-2 py-2 px-4";
    copyButton.style.opacity = '0.7';
    copyButton.style.border = 'none';
    copyButton.style.background = 'none';
    copyButton.style.color = '#fbfeff';
    copyButton.style.cursor = 'pointer';
    copyButton.onclick = function() { copyToClipboard('textarea-' + id); };

    container.appendChild(textarea);
    container.appendChild(copyButton);

    const likeButton = document.createElement('button');
    likeButton.innerHTML = '👍';
    likeButton.textContent = '👍';
    likeButton.className = 'like-btn';
    likeButton.addEventListener('click', function() {
    });
    container.appendChild(likeButton);

    const dislikeButton = document.createElement('button');
    dislikeButton.innerHTML = '👎';
    dislikeButton.textContent = '👎';
    dislikeButton.className = 'dislike-btn';
    dislikeButton.addEventListener('click', function() {
    });
    container.appendChild(dislikeButton);

    return { container, textarea };
}

    function sendToServer(text, reaction, headline) {
        var formData = new FormData();
        formData.append('text', text);
        formData.append('reaction', reaction);
        formData.append('headline', headline);

        // AJAX post request to save the data
        fetch('/save-csv/', {method: 'POST', body: formData})
            .then(response => response.json())
            .then(data => console.log(data))
            .catch((error) => {console.error('Error:', error);});
}

async function copyToClipboard(elementId) {
    const textarea = document.getElementById(elementId);
    if (!textarea) return;
    try {
        await navigator.clipboard.writeText(textarea.value);
    } catch (err) {
        console.error('Failed to copy text: ', err);
    }
}

let slider = document.getElementById('labels-range-input');
let tooltip = document.getElementById('slider-tooltip');

// Event listener for slider
slider.addEventListener('input', function() {
    // Update tooltip content with slider value
    tooltip.textContent = 'Wörter: ' + parseInt(this.value);
    // Calculate position for tooltip and position it
    let valuePercentage = (this.value - this.min) / (this.max - this.min);
    tooltip.style.left = `calc(${valuePercentage} * (100% - 1.2em))`;
    // Display tooltip
    tooltip.classList.remove('invisible');

    // Update form field 'words' with the slider value
    document.querySelector(`input[name='words']`).value = this.value;
});


