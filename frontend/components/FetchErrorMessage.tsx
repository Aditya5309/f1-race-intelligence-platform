/** Shared inline fetch-error banner — same markup/styling RaceCenterView
 * already used ad hoc, now reused by Driver Explorer, Compare Drivers, Team,
 * and Circuit Explorer (Task A2). Each caller supplies its own specific
 * message; this component only owns the presentation. */
export function FetchErrorMessage({ message }: { message: string }) {
  return <p className="text-sm text-[var(--color-accent-red)]">{message}</p>;
}
