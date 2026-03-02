import { useRef, useState, useEffect, useCallback } from 'react';
import './AudioPlayer.css';

interface AudioPlayerProps {
  src: string;
  seekTo?: number;
  onTimeUpdate?: (currentTime: number) => void;
  playbackRate?: number;
}

export default function AudioPlayer({
  src,
  seekTo,
  onTimeUpdate,
  playbackRate = 1,
}: AudioPlayerProps) {
  const audioRef = useRef<HTMLAudioElement>(null);
  const [playing, setPlaying] = useState(false);
  const [currentTime, setCurrentTime] = useState(0);
  const [duration, setDuration] = useState(0);

  // External seek control.
  // Note: if parent needs to re-seek to the same timestamp, it should
  // change the value slightly (e.g. seekTo + 0.001) or switch to a
  // callback pattern — React skips effects when deps are unchanged.
  useEffect(() => {
    if (seekTo !== undefined && audioRef.current) {
      audioRef.current.currentTime = seekTo;
    }
  }, [seekTo]);

  // Playback rate
  useEffect(() => {
    if (audioRef.current) {
      audioRef.current.playbackRate = playbackRate;
    }
  }, [playbackRate]);

  const handleTimeUpdate = useCallback(() => {
    if (!audioRef.current) return;
    const t = audioRef.current.currentTime;
    setCurrentTime(t);
    onTimeUpdate?.(t);
  }, [onTimeUpdate]);

  const handleLoadedMetadata = useCallback(() => {
    if (audioRef.current) {
      setDuration(audioRef.current.duration);
    }
  }, []);

  const togglePlay = useCallback(() => {
    if (!audioRef.current) return;
    if (playing) {
      audioRef.current.pause();
    } else {
      audioRef.current.play();
    }
    setPlaying(!playing);
  }, [playing]);

  const handleScrub = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    if (!audioRef.current) return;
    const t = parseFloat(e.target.value);
    audioRef.current.currentTime = t;
    setCurrentTime(t);
  }, []);

  const handleEnded = useCallback(() => {
    setPlaying(false);
  }, []);

  return (
    <div className="audio-player">
      <audio
        ref={audioRef}
        src={src}
        onTimeUpdate={handleTimeUpdate}
        onLoadedMetadata={handleLoadedMetadata}
        onEnded={handleEnded}
        preload="metadata"
      />
      <button className="audio-player-btn" onClick={togglePlay}>
        {playing ? 'Pause' : 'Play'}
      </button>
      <span className="audio-player-time">
        {formatTime(currentTime)} / {formatTime(duration)}
      </span>
      <input
        className="audio-player-scrub"
        type="range"
        min={0}
        max={duration || 0}
        step={0.1}
        value={currentTime}
        onChange={handleScrub}
      />
    </div>
  );
}

function formatTime(seconds: number): string {
  if (!isFinite(seconds)) return '0:00';
  const m = Math.floor(seconds / 60);
  const s = Math.floor(seconds % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}
