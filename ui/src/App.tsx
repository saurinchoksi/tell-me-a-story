import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import Layout from './components/Layout';
import Sessions from './pages/Sessions';
import SessionSpeakers from './pages/SessionSpeakers';
import ProfileGallery from './pages/ProfileGallery';
import ProfileDetail from './pages/ProfileDetail';
import ValidatorPage from './pages/ValidatorPage';
import './App.css';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        {/* Validator renders outside Layout — needs full viewport width */}
        <Route path="/sessions/:id/validate" element={<ValidatorPage />} />

        <Route element={<Layout />}>
          <Route path="/" element={<Navigate to="/sessions" replace />} />
          <Route path="/sessions" element={<Sessions />} />
          <Route path="/sessions/:id/speakers" element={<SessionSpeakers />} />
          <Route path="/profiles" element={<ProfileGallery />} />
          <Route path="/profiles/:id" element={<ProfileDetail />} />
        </Route>
      </Routes>
    </BrowserRouter>
  );
}
