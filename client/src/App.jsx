// src/App.js
import React from "react";
import "./App.css";
import ImageUpload from "./ImageUpload";

function App() {
  return (
    <div className="App">
      <header className="App-header">
        <h1>Tomato leaf disease detection</h1>
        <ImageUpload />
      </header>
    </div>
  );
}

export default App;
