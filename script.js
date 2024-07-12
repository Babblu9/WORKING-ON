document.getElementById('try-on-button').addEventListener('click', function() {
    const userImage = document.getElementById('user-image-upload').files[0];
    const clothImage = document.getElementById('cloth-image-upload').files[0];
    const category = document.getElementById('category').value;
    const caption = document.getElementById('caption').value;

    if (!userImage || !clothImage) {
        alert('Please upload both user and cloth images.');
        return;
    }

    const formData = new FormData();
    formData.append('user_image', userImage);
    formData.append('cloth_image', clothImage);
    formData.append('category', category);
    formData.append('caption', caption);

    fetch('/try-on', {
        method: 'POST',
        body: formData
    })
    .then(response => response.blob())
    .then(blob => {
        const url = URL.createObjectURL(blob);
        const resultImg = document.createElement('img');
        resultImg.src = url;
        document.getElementById('result-container').innerHTML = '';
        document.getElementById('result-container').appendChild(resultImg);
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the Try-On.');
    });
});
