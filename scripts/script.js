document.getElementById("imageInput").addEventListener("change", function(event) {
    const file = event.target.files[0];
    if (file && file.type.startsWith("image/")) {
        const reader = new FileReader();
        reader.onload = function(e) {
            const img = document.getElementById("displayImage");
            img.src = e.target.result;
            img.style.display = "block"; // Show the image once loaded
        };
        reader.readAsDataURL(file);
    } else {
        alert("Please upload a valid image file.");
    }
});
