/**
 * WaveformPlayer — WaveSurfer wrapper with minimap plugin.
 *
 * Manages the WaveSurfer lifecycle: creates on mount and fires a 60fps
 * onTimeUpdate via requestAnimationFrame for smooth active segment tracking.
 */

import { useEffect, useRef, useCallback, useImperativeHandle, forwardRef } from 'react';
import WaveSurfer from 'wavesurfer.js';
import Minimap from 'wavesurfer.js/dist/plugins/minimap.esm.js';
import './WaveformPlayer.css';

export interface WaveformPlayerHandle {
  play(): void;
  pause(): void;
  playPause(): void;
  setTime(seconds: number): void;
  getCurrentTime(): number;
  setPlaybackRate(rate: number): void;
  isPlaying(): boolean;
}

interface WaveformPlayerProps {
  audioUrl: string;
  onTimeUpdate: (time: number) => void;
  onReady?: (duration: number) => void;
  onPlayPause?: (playing: boolean) => void;
  playbackRate?: number;
}

function getCSSColor(prop: string, fallback: string): string {
  const val = getComputedStyle(document.documentElement).getPropertyValue(prop).trim();
  return val || fallback;
}

const WaveformPlayer = forwardRef<WaveformPlayerHandle, WaveformPlayerProps>(
  function WaveformPlayer({ audioUrl, onTimeUpdate, onReady, onPlayPause, playbackRate = 1 }, ref) {
    const containerRef = useRef<HTMLDivElement>(null);
    const minimapRef = useRef<HTMLDivElement>(null);
    const wsRef = useRef<WaveSurfer | null>(null);
    const rafRef = useRef<number>(0);
    const savedTimeRef = useRef(0);
    const savedRateRef = useRef(playbackRate);

    // Keep callback refs stable so WaveSurfer events always call latest versions
    const onTimeUpdateRef = useRef(onTimeUpdate);
    const onReadyRef = useRef(onReady);
    const onPlayPauseRef = useRef(onPlayPause);
    useEffect(() => { onTimeUpdateRef.current = onTimeUpdate; }, [onTimeUpdate]);
    useEffect(() => { onReadyRef.current = onReady; }, [onReady]);
    useEffect(() => { onPlayPauseRef.current = onPlayPause; }, [onPlayPause]);

    const startRAF = useCallback(() => {
      const tick = () => {
        if (wsRef.current) {
          onTimeUpdateRef.current(wsRef.current.getCurrentTime());
        }
        rafRef.current = requestAnimationFrame(tick);
      };
      rafRef.current = requestAnimationFrame(tick);
    }, []);

    const stopRAF = useCallback(() => {
      if (rafRef.current) {
        cancelAnimationFrame(rafRef.current);
        rafRef.current = 0;
      }
    }, []);

    // Create/recreate WaveSurfer when audioUrl changes
    useEffect(() => {
      if (!containerRef.current || !minimapRef.current) return;

      // Save position before destroying
      if (wsRef.current) {
        savedTimeRef.current = wsRef.current.getCurrentTime();
        savedRateRef.current = wsRef.current.getPlaybackRate();
        stopRAF();
        wsRef.current.destroy();
      }

      const waveColor = getCSSColor('--waveform-color', '#999');
      const progressColor = getCSSColor('--waveform-progress', '#BFA265');
      const cursorColor = getCSSColor('--waveform-cursor', '#6A3843');

      const ws = WaveSurfer.create({
        container: containerRef.current,
        url: audioUrl,
        waveColor,
        progressColor,
        cursorColor,
        cursorWidth: 2,
        height: 100,
        minPxPerSec: 100,
        normalize: true,
        plugins: [
          Minimap.create({
            container: minimapRef.current,
            height: 36,
            waveColor: getCSSColor('--waveform-minimap', '#ccc'),
            progressColor,
            cursorColor,
            normalize: true,
          }),
        ],
      });

      ws.on('ready', () => {
        ws.setPlaybackRate(savedRateRef.current);
        if (savedTimeRef.current > 0) {
          ws.setTime(savedTimeRef.current);
        }
        onReadyRef.current?.(ws.getDuration());
      });

      ws.on('play', () => {
        startRAF();
        onPlayPauseRef.current?.(true);
      });

      ws.on('pause', () => {
        stopRAF();
        // Fire one last time update at pause position
        onTimeUpdateRef.current(ws.getCurrentTime());
        onPlayPauseRef.current?.(false);
      });

      ws.on('seeking', () => {
        onTimeUpdateRef.current(ws.getCurrentTime());
      });

      wsRef.current = ws;

      return () => {
        stopRAF();
        ws.destroy();
        wsRef.current = null;
      };
    }, [audioUrl, startRAF, stopRAF]);

    // Update playback rate without recreating
    useEffect(() => {
      if (wsRef.current) {
        wsRef.current.setPlaybackRate(playbackRate);
        savedRateRef.current = playbackRate;
      }
    }, [playbackRate]);

    useImperativeHandle(ref, () => ({
      play() { wsRef.current?.play(); },
      pause() { wsRef.current?.pause(); },
      playPause() { wsRef.current?.playPause(); },
      setTime(seconds: number) {
        if (wsRef.current) {
          wsRef.current.setTime(seconds);
          onTimeUpdateRef.current(seconds);
        }
      },
      getCurrentTime() { return wsRef.current?.getCurrentTime() ?? 0; },
      setPlaybackRate(rate: number) { wsRef.current?.setPlaybackRate(rate); },
      isPlaying() { return wsRef.current?.isPlaying() ?? false; },
    }));

    return (
      <div className="waveform-player">
        <div ref={minimapRef} className="waveform-minimap" />
        <div ref={containerRef} className="waveform-main" />
      </div>
    );
  },
);

export default WaveformPlayer;
