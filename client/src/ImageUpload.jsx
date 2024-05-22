import React, { useState } from "react";
import axios from "axios";
import "./App.css";

const ImageUpload = () => {
  const [file, setFile] = useState(null);
  const [message, setMessage] = useState("");
  const [imageUrl, setImageUrl] = useState("");

  const handleFileChange = (event) => {
    setFile(event.target.files[0]);
  };

  const handleSubmit = async (event) => {
    event.preventDefault();
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await axios.post(
        "http://127.0.0.1:8000/upload-image/",
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      setMessage(`File uploaded successfully: ${response.data.filename}`);
      setImageUrl(URL.createObjectURL(file)); // Update the image URL to the uploaded file
    } catch (error) {
      setMessage("Error uploading file");
    }
  };

  return (
    <div className="upload-container">
      <form onSubmit={handleSubmit}>
        <input type="file" id="file-input" onChange={handleFileChange} />
        <span>
          <label htmlFor="file-input">Choose File</label>
          <button type="submit">Upload</button>
        </span>
      </form>
      {message && <p className="message">{message}</p>}
      {imageUrl && (
        <img src={imageUrl} alt="Uploaded" className="uploaded-image" />
      )}
    </div>
  );
};

export default ImageUpload;
