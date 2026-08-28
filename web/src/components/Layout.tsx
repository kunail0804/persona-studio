import { Link, Outlet } from "react-router-dom";

export function Layout() {
  return (
    <div className="min-h-screen bg-neutral-950 text-neutral-100">
      <header className="flex items-center justify-between border-b border-neutral-800 px-6 py-4">
        <Link to="/" className="text-lg font-semibold tracking-tight">
          Persona Studio
        </Link>
        <Link to="/settings" className="text-sm text-neutral-400 hover:text-neutral-100">
          Paramètres
        </Link>
      </header>
      <main className="mx-auto max-w-3xl px-6 py-8">
        <Outlet />
      </main>
    </div>
  );
}
