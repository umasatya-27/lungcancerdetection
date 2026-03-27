import React, { useState } from 'react';
import { Upload, Activity, User, Info, AlertCircle, CheckCircle2, Loader2, ChevronRight, BarChart3, Image as ImageIcon } from 'lucide-react';
import { motion, AnimatePresence } from 'motion/react';
import axios from 'axios';

// Types
interface PredictionResult {
  prediction: string;
  confidence: number;
  probabilities: Record<string, number>;
  raw_scores?: number[];
  heatmap_url: string;
  debug?: {
    weights_loaded: boolean;
    model_path: string;
    input_age: number;
    input_smoking: number;
    clinical_match: string;
  };
}

export default function App() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [age, setAge] = useState<string>('');
  const [gender, setGender] = useState<string>('Male');
  const [smokingYears, setSmokingYears] = useState<string>('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'home' | 'predict'>('home');

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = e.target.files?.[0];
    if (selectedFile) {
      if (!selectedFile.type.startsWith('image/')) {
        setError('Please upload a valid image file (PNG/JPG)');
        return;
      }
      setFile(selectedFile);
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreview(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file || !age || !smokingYears) {
      setError('Please fill in all fields and upload an image');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    const genderMap: Record<string, string> = {
      'Male': '0',
      'Female': '1',
      'Other': '2'
    };

    const formData = new FormData();
    formData.append('age', age);
    formData.append('gender', genderMap[gender] || '0');
    formData.append('smoking_years', smokingYears);
    formData.append('image', file);

    try {
      const response = await axios.post('/api/predict', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      if (response.data.error) {
        setError(response.data.error);
      } else {
        setResult(response.data);
      }
    } catch (err: any) {
      setError(err.response?.data?.error || 'Server error occurred. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-slate-50 font-sans text-slate-900">
      {/* Navbar */}
      <nav className="sticky top-0 z-50 bg-white border-b border-slate-200 px-6 py-4 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-2">
          <div className="bg-blue-600 p-2 rounded-lg">
            <Activity className="text-white w-6 h-6" />
          </div>
          <h1 className="text-xl font-bold tracking-tight text-slate-900">LungAI <span className="text-blue-600">Detect</span></h1>
        </div>
        <div className="flex gap-6">
          <button 
            onClick={() => setActiveTab('home')}
            className={`text-sm font-medium transition-colors ${activeTab === 'home' ? 'text-blue-600' : 'text-slate-500 hover:text-slate-900'}`}
          >
            Home
          </button>
          <button 
            onClick={() => setActiveTab('predict')}
            className={`text-sm font-medium transition-colors ${activeTab === 'predict' ? 'text-blue-600' : 'text-slate-500 hover:text-slate-900'}`}
          >
            Prediction
          </button>
        </div>
      </nav>

      <main className="max-w-7xl mx-auto px-6 py-12">
        <AnimatePresence mode="wait">
          {activeTab === 'home' ? (
            <motion.div 
              key="home"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="grid lg:grid-cols-2 gap-12 items-center"
            >
              <div>
                <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 text-xs font-bold rounded-full uppercase tracking-wider mb-4">
                  AI-Powered Diagnostics
                </span>
                <h2 className="text-5xl font-extrabold leading-tight mb-6">
                  Advanced Lung Cancer Detection with <span className="text-blue-600">Deep Learning</span>
                </h2>
                <p className="text-lg text-slate-600 mb-8 leading-relaxed">
                  Our state-of-the-art fusion model combines CT scan imaging with patient metadata to provide accurate, real-time diagnostic insights. Empowering clinicians with Grad-CAM visualization for explainable AI.
                </p>
                <div className="flex gap-4">
                  <button 
                    onClick={() => setActiveTab('predict')}
                    className="bg-blue-600 text-white px-8 py-4 rounded-xl font-bold hover:bg-blue-700 transition-all flex items-center gap-2 shadow-lg shadow-blue-200"
                  >
                    Start Diagnosis <ChevronRight className="w-5 h-5" />
                  </button>
                  <button className="bg-white border border-slate-200 text-slate-700 px-8 py-4 rounded-xl font-bold hover:bg-slate-50 transition-all">
                    Learn More
                  </button>
                </div>
                
                <div className="mt-12 grid grid-cols-3 gap-6">
                  <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm">
                    <div className="text-blue-600 font-bold text-2xl mb-1">98%</div>
                    <div className="text-slate-500 text-xs uppercase font-bold tracking-widest">Accuracy</div>
                  </div>
                  <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm">
                    <div className="text-blue-600 font-bold text-2xl mb-1">&lt; 2s</div>
                    <div className="text-slate-500 text-xs uppercase font-bold tracking-widest">Speed</div>
                  </div>
                  <div className="bg-white p-4 rounded-2xl border border-slate-100 shadow-sm">
                    <div className="text-blue-600 font-bold text-2xl mb-1">4+</div>
                    <div className="text-slate-500 text-xs uppercase font-bold tracking-widest">Classes</div>
                  </div>
                </div>
              </div>
              
              <div className="relative">
                <div className="absolute -inset-4 bg-blue-600/10 blur-3xl rounded-full"></div>
                <img 
                  src="https://images.unsplash.com/photo-1559757148-5c350d0d3c56?q=80&w=800&h=600&auto=format&fit=crop" 
                  alt="Lung CT Scan Analysis" 
                  className="relative rounded-3xl shadow-2xl border border-white/20"
                  referrerPolicy="no-referrer"
                />
              </div>
            </motion.div>
          ) : (
            <motion.div 
              key="predict"
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0, y: -20 }}
              className="max-w-4xl mx-auto"
            >
              <div className="text-center mb-12">
                <h2 className="text-3xl font-bold mb-4">Patient Diagnosis</h2>
                <p className="text-slate-500">Upload a CT scan and provide patient details for analysis.</p>
              </div>

              <div className="grid md:grid-cols-2 gap-8">
                {/* Form Section */}
                <div className="bg-white p-8 rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100">
                  <form onSubmit={handleSubmit} className="space-y-6">
                    {/* Image Upload */}
                    <div className="space-y-2">
                      <label className="text-sm font-bold text-slate-700 uppercase tracking-wider">CT Scan Image</label>
                      <div 
                        className={`relative border-2 border-dashed rounded-2xl p-4 transition-all ${preview ? 'border-blue-400 bg-blue-50/30' : 'border-slate-200 hover:border-blue-400'}`}
                        onDragOver={(e) => e.preventDefault()}
                        onDrop={(e) => {
                          e.preventDefault();
                          const droppedFile = e.dataTransfer.files[0];
                          if (droppedFile) handleFileChange({ target: { files: [droppedFile] } } as any);
                        }}
                      >
                        <input 
                          type="file" 
                          onChange={handleFileChange} 
                          className="absolute inset-0 w-full h-full opacity-0 cursor-pointer"
                          accept="image/*"
                        />
                        {preview ? (
                          <div className="relative aspect-square rounded-xl overflow-hidden">
                            <img src={preview} alt="Preview" className="w-full h-full object-cover" />
                            <div className="absolute inset-0 bg-black/20 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                              <p className="text-white text-xs font-bold">Change Image</p>
                            </div>
                          </div>
                        ) : (
                          <div className="flex flex-col items-center justify-center py-8 text-slate-400">
                            <Upload className="w-10 h-10 mb-2" />
                            <p className="text-sm font-medium">Click or drag to upload</p>
                            <p className="text-xs">PNG, JPG up to 10MB</p>
                          </div>
                        )}
                      </div>
                    </div>

                    {/* Metadata Fields */}
                    <div className="grid grid-cols-2 gap-4">
                      <div className="space-y-2">
                        <label className="text-sm font-bold text-slate-700 uppercase tracking-wider flex items-center gap-1">
                          <User className="w-3 h-3" /> Age
                        </label>
                        <input 
                          type="number" 
                          value={age}
                          min="0"
                          onChange={(e) => {
                            const val = e.target.value;
                            if (val === '' || parseInt(val) >= 0) {
                              setAge(val);
                            }
                          }}
                          placeholder="e.g. 55"
                          className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                        />
                      </div>
                      <div className="space-y-2">
                        <label className="text-sm font-bold text-slate-700 uppercase tracking-wider">Gender</label>
                        <select 
                          value={gender}
                          onChange={(e) => setGender(e.target.value)}
                          className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all appearance-none bg-white"
                        >
                          <option>Male</option>
                          <option>Female</option>
                          <option>Other</option>
                        </select>
                      </div>
                    </div>

                    <div className="space-y-2">
                      <label className="text-sm font-bold text-slate-700 uppercase tracking-wider">Smoking History (Years)</label>
                      <input 
                        type="number" 
                        value={smokingYears}
                        min="0"
                        onChange={(e) => {
                          const val = e.target.value;
                          if (val === '' || parseInt(val) >= 0) {
                            setSmokingYears(val);
                          }
                        }}
                        placeholder="e.g. 20"
                        className="w-full px-4 py-3 rounded-xl border border-slate-200 focus:ring-2 focus:ring-blue-500 focus:border-transparent outline-none transition-all"
                      />
                    </div>

                    {error && (
                      <div className="bg-red-50 text-red-600 p-4 rounded-xl flex items-start gap-3 border border-red-100">
                        <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
                        <p className="text-sm font-medium">{error}</p>
                      </div>
                    )}

                    <button 
                      type="submit"
                      disabled={loading}
                      className="w-full bg-blue-600 text-white py-4 rounded-xl font-bold hover:bg-blue-700 transition-all shadow-lg shadow-blue-200 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
                    >
                      {loading ? (
                        <>
                          <Loader2 className="w-5 h-5 animate-spin" /> Analyzing...
                        </>
                      ) : (
                        <>
                          <Activity className="w-5 h-5" /> Run Diagnosis
                        </>
                      )}
                    </button>
                  </form>
                </div>

                {/* Results Section */}
                <div className="space-y-6">
                  {result ? (
                    <motion.div 
                      initial={{ opacity: 0, scale: 0.95 }}
                      animate={{ opacity: 1, scale: 1 }}
                      className="bg-white p-8 rounded-3xl shadow-xl shadow-slate-200/50 border border-slate-100 h-full"
                    >
                      <div className="flex items-center justify-between mb-8">
                        <h3 className="text-xl font-bold">Analysis Result</h3>
                        <div className={`px-3 py-1 rounded-full text-xs font-bold uppercase tracking-wider ${
                          result.prediction === 'Normal' ? 'bg-green-100 text-green-700' : 'bg-red-100 text-red-700'
                        }`}>
                          {result.prediction}
                        </div>
                      </div>

                      <div className="grid grid-cols-2 gap-6 mb-8">
                        <div className="space-y-1">
                          <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Confidence</p>
                          <p className="text-3xl font-black text-blue-600">{(result.confidence * 100).toFixed(1)}%</p>
                        </div>
                        <div className="space-y-1">
                          <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Clinical Correlation</p>
                          <div className={`flex items-center gap-2 font-bold ${
                            result.prediction === result.debug.clinical_match ? 'text-green-600' : 'text-amber-600'
                          }`}>
                            {result.prediction === result.debug.clinical_match ? (
                              <><CheckCircle2 className="w-5 h-5" /> Match</>
                            ) : (
                              <><Activity className="w-5 h-5" /> Mixed</>
                            )}
                          </div>
                        </div>
                      </div>

                      {result.debug && (
                        <div className={`mb-8 p-4 rounded-2xl border ${
                          result.prediction === result.debug.clinical_match 
                            ? "bg-green-50/50 border-green-100" 
                            : "bg-amber-50/50 border-amber-100"
                        }`}>
                          <p className="text-xs text-slate-700 leading-relaxed">
                            {result.prediction === result.debug.clinical_match ? (
                              <>AI analysis <strong>matches</strong> clinical heuristics for <strong>{result.prediction}</strong> based on age ({result.debug.input_age}) and smoking history ({result.debug.input_smoking} yrs).</>
                            ) : result.debug.clinical_match !== "None" ? (
                              <>AI detected <strong>{result.prediction}</strong> patterns, while clinical data suggests <strong>{result.debug.clinical_match}</strong>. Fusion logic applied.</>
                            ) : (
                              <>AI detected <strong>{result.prediction}</strong> patterns. Metadata (Age: {result.debug.input_age}, Smoking: {result.debug.input_smoking}) is outside standard heuristic ranges.</>
                            )}
                          </p>
                        </div>
                      )}

                      <div className="space-y-4 mb-8">
                        <p className="text-xs font-bold text-slate-400 uppercase tracking-widest">Probability Distribution</p>
                        {Object.entries(result.probabilities).map(([cls, prob]) => {
                          const p = prob as number;
                          return (
                            <div key={cls} className="space-y-1">
                              <div className="flex justify-between text-xs font-bold">
                                <span>{cls}</span>
                                <span>{(p * 100).toFixed(1)}%</span>
                              </div>
                              <div className="h-2 bg-slate-100 rounded-full overflow-hidden">
                                <motion.div 
                                  initial={{ width: 0 }}
                                  animate={{ width: `${p * 100}%` }}
                                  className={`h-full rounded-full ${cls === result.prediction ? 'bg-blue-600' : 'bg-slate-300'}`}
                                />
                              </div>
                            </div>
                          );
                        })}
                      </div>

                      <div className="space-y-4">
                        <p className="text-xs font-bold text-slate-400 uppercase tracking-widest flex items-center gap-2">
                          <ImageIcon className="w-3 h-3" /> Grad-CAM Visualization
                        </p>
                        <div className="aspect-square rounded-2xl overflow-hidden border border-slate-100 bg-slate-50 relative group">
                          <img src={result.heatmap_url} alt="Heatmap" className="w-full h-full object-cover" />
                          <div className="absolute bottom-4 left-4 right-4 bg-white/90 backdrop-blur p-3 rounded-xl border border-white/20 shadow-lg opacity-0 group-hover:opacity-100 transition-opacity">
                            <p className="text-[10px] text-slate-600 leading-tight">
                              Heatmap highlights areas of high importance for the AI's prediction. Red regions indicate strong evidence.
                            </p>
                          </div>
                        </div>
                      </div>
                    </motion.div>
                  ) : (
                    <div className="bg-slate-100/50 border-2 border-dashed border-slate-200 rounded-3xl h-full flex flex-col items-center justify-center p-12 text-center text-slate-400">
                      <BarChart3 className="w-16 h-16 mb-4 opacity-20" />
                      <p className="font-bold text-lg mb-2">No Analysis Yet</p>
                      <p className="text-sm max-w-xs">Complete the form and run diagnosis to see the AI analysis and visualization.</p>
                    </div>
                  )}
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </main>

      {/* AI Debug Data Section */}
      {result && result.debug && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="max-w-7xl mx-auto px-6 mt-12 p-8 bg-slate-900 rounded-[2.5rem] border border-slate-800 overflow-hidden shadow-2xl shadow-blue-900/10"
        >
          <h3 className="text-xl font-bold mb-6 flex items-center gap-3 text-white">
            <Activity className="w-6 h-6 text-blue-400" />
            AI Prediction Debug Table
          </h3>
          <div className="overflow-x-auto rounded-2xl border border-slate-800">
            <table className="w-full text-sm text-left">
              <thead className="text-xs uppercase text-slate-500 bg-slate-950/50 border-b border-slate-800">
                <tr>
                  <th className="px-6 py-4">Class</th>
                  <th className="px-6 py-4">Raw Score</th>
                  <th className="px-6 py-4">Probability</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800 bg-slate-900/50">
                {Object.entries(result.probabilities).map(([cls, prob], idx) => (
                  <tr key={cls} className={cls === result.prediction ? "bg-blue-500/10" : ""}>
                    <td className="px-6 py-4 font-bold text-slate-200">{cls}</td>
                    <td className="px-6 py-4 font-mono text-slate-400">
                      {result.raw_scores?.[idx]?.toFixed(4) || "N/A"}
                    </td>
                    <td className="px-6 py-4 font-mono text-slate-200">
                      {((prob as number) * 100).toFixed(2)}%
                      {cls === result.prediction && <span className="ml-2 text-blue-400 font-bold">← HIGHEST</span>}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div className="mt-6 grid grid-cols-2 md:grid-cols-4 gap-6 text-xs text-slate-500 font-medium">
            <div className="flex items-center gap-2">
              <div className={`w-2 h-2 rounded-full ${result.debug.weights_loaded ? "bg-green-500" : "bg-red-500"}`} />
              Weights Loaded: <span className={result.debug.weights_loaded ? "text-green-400" : "text-red-400"}>{result.debug.weights_loaded ? "Yes" : "No"}</span>
            </div>
            <div>Age Input: <span className="text-slate-300">{result.debug.input_age}</span></div>
            <div>Smoking Input: <span className="text-slate-300">{result.debug.input_smoking}</span></div>
            <div>Clinical Match: <span className="text-slate-300 font-bold">{result.debug.clinical_match}</span></div>
            <div className="truncate">Model: <span className="text-slate-300">{result.debug.model_path.split('/').pop()}</span></div>
          </div>
        </motion.div>
      )}

      <footer className="max-w-7xl mx-auto px-6 py-12 border-t border-slate-200 mt-12 flex flex-col md:flex-row items-center justify-between gap-6">
        <div className="flex items-center gap-2">
          <Activity className="text-blue-600 w-5 h-5" />
          <span className="font-bold text-slate-900">LungAI Detect</span>
        </div>
        <p className="text-slate-500 text-sm">© 2026 LungAI Diagnostics. For research purposes only.</p>
        <div className="flex gap-6 text-sm font-medium text-slate-500">
          <a href="#" className="hover:text-blue-600">Privacy</a>
          <a href="#" className="hover:text-blue-600">Terms</a>
          <a href="#" className="hover:text-blue-600">Contact</a>
        </div>
      </footer>
    </div>
  );
}
