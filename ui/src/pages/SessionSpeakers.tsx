import { useParams } from 'react-router-dom';

/** Placeholder for Task 6 — Session speaker review UI. */
export default function SessionSpeakers() {
  const { id } = useParams<{ id: string }>();

  return (
    <div>
      <h1>Speaker Review</h1>
      <p>Session: <code>{id}</code></p>
      <p>Task 6 will build the speaker review interface here.</p>
    </div>
  );
}
