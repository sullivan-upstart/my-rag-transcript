import React, { useState } from 'react';
import axios from 'axios';

export default function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer] = useState('');
  const [status, setStatus] = useState('');

  const upload = async () => {
    if (!file) return;
    const data = new FormData();
    data.append('file', file);
    try {
      const res = await axios.post('http://localhost:8000/upload', data);
      setStatus(`Uploaded ${res.data.filename}`);
    } catch (err) {
      setStatus('Upload failed');
    }
  };

  const ask = async () => {
    if (!question.trim()) return;
    const data = new FormData(); data.append('question', question);
    const { data: res } = await axios.post('http://localhost:8000/query', data);
    setAnswer(res.answer);
  };

  return (
    <div className="app">
      <h1>RAG Transcript QA</h1>

      <section>
        <h2>1️⃣ Upload transcript</h2>
        <input
          type="file"
          accept=".txt,.vtt,.srt"
          onChange={e => setFile(e.target.files[0])}
        />
        <button onClick={upload} style={{ marginLeft: 10 }}>Upload</button>
        {status && (
          <p
            className={`status ${status.startsWith('Uploaded') ? 'success' : 'error'}`}
          >
            {status}
          </p>
        )}
      </section>

      <section>
        <h2>2️⃣ Ask a question</h2>
        <input
          className="question-input"
          value={question}
          onChange={e => setQuestion(e.target.value)}
          placeholder="e.g. What service handles auth?"
        />
        <button onClick={ask} style={{ marginLeft: 10 }}>Ask</button>
      </section>

      <section>
        <h2>Answer</h2>
        <pre>{answer}</pre>
      </section>
    </div>
  );
}

