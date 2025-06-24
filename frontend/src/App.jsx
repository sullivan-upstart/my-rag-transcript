import React, { useState } from 'react';
import axios from 'axios';

export default function App() {
  const [file, setFile] = useState(null);
  const [question, setQuestion] = useState('');
  const [answer, setAnswer]   = useState('');

  const upload = async () => {
    if (!file) return;
    const data = new FormData(); data.append('file', file);
    await axios.post('http://localhost:8000/upload', data);
    alert('Uploaded!');
  };

  const ask = async () => {
    if (!question.trim()) return;
    const data = new FormData(); data.append('question', question);
    const { data: res } = await axios.post('http://localhost:8000/query', data);
    setAnswer(res.answer);
  };

  return (
    <main style={{fontFamily:'sans-serif',padding:20}}>
      <h1>RAG Transcript QA</h1>

      <section>
        <h2>1️⃣ Upload transcript</h2>
        <input type="file" accept=".txt,.vtt,.srt"
               onChange={e=>setFile(e.target.files[0])}/>
        <button onClick={upload} style={{marginLeft:10}}>Upload</button>
      </section>

      <section style={{marginTop:30}}>
        <h2>2️⃣ Ask a question</h2>
        <input style={{width:300}}
               value={question}
               onChange={e=>setQuestion(e.target.value)}
               placeholder="e.g. What service handles auth?"/>
        <button onClick={ask} style={{marginLeft:10}}>Ask</button>
      </section>

      <section style={{marginTop:30}}>
        <h2>Answer</h2>
        <pre style={{background:'#f6f8fa',padding:10}}>{answer}</pre>
      </section>
    </main>
  );
}
