function showInput(type) {
    document.getElementById('urlInput').classList.add('hidden');
    document.getElementById('fileInput').classList.add('hidden');
    document.getElementById(type + 'Input').classList.remove('hidden');
}

async function submitUrl() {
    const url = document.getElementById('url').value;
    const response = await fetch('/generate_summary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url: url }),
    });
    const result = await response.json();
    if (result.summary) {
        document.getElementById('summaryOutput').classList.remove('hidden');
        document.getElementById('summary').value = result.summary;
        document.getElementById('hyperlinkInput').classList.remove('hidden');

        const hyperlinks = result.hyperlinks;
        const hyperlinkSelect = document.getElementById('hyperlinkSelect');
        hyperlinkSelect.innerHTML = '';
        hyperlinks.forEach((link, index) => {
            const option = document.createElement('option');
            option.value = index;
            option.textContent = link;
            hyperlinkSelect.appendChild(option);
        });
        const hyperlinksInput = document.getElementById('hyperlinks');
        if (hyperlinksInput) {
            hyperlinksInput.value = JSON.stringify(hyperlinks);
        }
    } else {
        alert('Error: ' + result.error);
    }
}

async function submitHyperlink() {
    const selectedLinkIndex = document.getElementById('hyperlinkSelect').value;
    const hyperlinksInput= document.getElementById('hyperlinks');

    if(!hyperlinksInput){
        console.error('Element with id "hyperlinks" not found');
        return;
    }

    const hyperlinks = JSON.parse(hyperlinksInput.value);

    if (!hyperlinks || !Array.isArray(hyperlinks)) {
        console.error('Invalid hyperlinks data.');
        return;
    }

    const url_hp = hyperlinks[selectedLinkIndex];
    console.log('Selected URL:', url_hp);

    const response = await fetch('/generate_summary', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ url_hp: url_hp }),
    });
    const result = await response.json();
    console.log('Resposne: ',result);
    if (result.summary) {
        document.getElementById('summaryOutput').classList.remove('hidden');
        document.getElementById('summary').value = result.summary;
        document.getElementById('questionInput').classList.remove('hidden');
    } else {
        alert('Error: ' + result.error);
    }
}

async function submitFile() {
    const fileInput = document.getElementById('file');
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);
    
    const response = await fetch('/generate_summary', {
        method: 'POST',
        body: formData
    });
    const result = await response.json();
    if (result.summary) {
        document.getElementById('summaryOutput').classList.remove('hidden');
        document.getElementById('summary').value = result.summary;
        document.getElementById('questionInput').classList.remove('hidden');
    } else {
        alert('Error: ' + result.error);
    }
}

async function submitQuestion() {
    const question = document.getElementById('question').value;
    const response = await fetch('/find_answer', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: question }),
    });
    const result = await response.json();
    if (result.answer) {
        document.getElementById('answerOutput').classList.remove('hidden');
        document.getElementById('answer').value = result.answer;
    } else {
        alert('Error: ' + result.error);
    }
}
