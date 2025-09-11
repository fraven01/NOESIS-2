import React, { useEffect, useRef, useState } from 'react';
import { Menu, Moon, Search, Sun, User } from 'lucide-react';

/**
 * AppShell stellt den grundlegenden Seitenrahmen bereit.
 * Header, Sidebar, Main und Footer werden verwaltet.
 *
 * ```tsx
 * <AppShell>
 *   <div>Inhalt</div>
 * </AppShell>
 * ```
 */
export interface AppShellProps {
  children: React.ReactNode;
}

export const AppShell: React.FC<AppShellProps> = ({ children }) => {
  const [open, setOpen] = useState(false);
  const toggleRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open && toggleRef.current) {
      toggleRef.current.focus();
    }
  }, [open]);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const navItems = [
    'Dashboard',
    'Projekte',
    'Dokumente',
    'Workflows',
    'Berichte',
    'Einstellungen',
  ];

  const Sidebar = (
    <nav className="p-4 space-y-2" aria-label="Hauptnavigation">
      {navItems.map((item) => (
        <a
          key={item}
          href="#"
          className="block rounded px-2 py-1 hover:bg-accent focus:outline-none focus-visible:ring"
        >
          {item}
        </a>
      ))}
    </nav>
  );

  return (
    <div className="grid min-h-screen md:grid-cols-[240px_1fr]">
      <aside className="hidden md:block border-r bg-bg">{Sidebar}</aside>

      {open && (
        <div
          role="dialog"
          aria-modal="true"
          className="fixed inset-0 z-50 flex md:hidden"
        >
          <div className="w-60 bg-bg border-r overflow-y-auto">{Sidebar}</div>
          <button
            aria-label="Close sidebar"
            className="flex-1 bg-black/50"
            onClick={() => setOpen(false)}
          />
        </div>
      )}

      <div className="flex flex-col">
        <header className="sticky top-0 z-40 border-b bg-bg">
          <div className="flex h-14 items-center gap-2 px-4">
            <button
              ref={toggleRef}
              aria-label="Open sidebar"
              className="-ml-1 rounded p-2 hover:bg-accent focus:outline-none focus-visible:ring md:hidden"
              onClick={() => setOpen(true)}
            >
              <Menu className="h-5 w-5" />
            </button>
            <form className="flex-1">
              <label htmlFor="search" className="sr-only">
                Suche
              </label>
              <div className="relative">
                <Search className="pointer-events-none absolute left-2 top-1/2 h-4 w-4 -translate-y-1/2 text-muted" />
                <input
                  id="search"
                  type="search"
                  placeholder="Suche..."
                  className="w-full rounded border py-1 pl-8 pr-2 focus:outline-none focus-visible:ring"
                />
              </div>
            </form>
            <button
              aria-label="Toggle theme"
              className="rounded p-2 hover:bg-accent focus:outline-none focus-visible:ring"
              onClick={() => document.documentElement.classList.toggle('dark')}
            >
              <Sun className="h-5 w-5 dark:hidden" />
              <Moon className="hidden h-5 w-5 dark:block" />
            </button>
            <button
              aria-label="User menu"
              className="rounded p-2 hover:bg-accent focus:outline-none focus-visible:ring"
            >
              <User className="h-5 w-5" />
            </button>
          </div>
        </header>
        <main className="flex-1">
          <div className="mx-auto max-w-7xl p-4">
            <div className="mb-4 flex items-center justify-between">
              <nav aria-label="Breadcrumb">
                <ol className="flex gap-2 text-sm">
                  <li>
                    <a href="#" className="text-muted hover:text-fg">
                      Home
                    </a>
                  </li>
                  <li aria-hidden="true">/</li>
                  <li className="text-fg">Aktuell</li>
                </ol>
              </nav>
              <div className="flex gap-2">
                <button className="rounded bg-accent px-3 py-1 text-sm focus:outline-none focus-visible:ring">
                  Aktion
                </button>
              </div>
            </div>
            {children}
          </div>
        </main>
        <footer className="border-t p-4 text-sm text-muted">
          Build: {process.env.BUILD || 'dev'}
        </footer>
      </div>
    </div>
  );
};

export default AppShell;
