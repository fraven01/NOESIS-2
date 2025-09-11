import React from 'react';

/**
 * Application shell layout with sticky header, sidebar navigation, main content and footer.
 * Usage:
 * <AppShell breadcrumbs={<Breadcrumbs/>} actions={<Actions/>} buildInfo="1.0">
 *   <Content />
 * </AppShell>
 */
export interface AppShellProps {
  breadcrumbs?: React.ReactNode;
  actions?: React.ReactNode;
  children: React.ReactNode;
  buildInfo?: string;
}

export function AppShell({ breadcrumbs, actions, children, buildInfo }: AppShellProps) {
  const [open, setOpen] = React.useState(false);
  const triggerRef = React.useRef<HTMLButtonElement>(null);

  React.useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        setOpen(false);
        triggerRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const close = () => {
    setOpen(false);
    triggerRef.current?.focus();
  };

  const nav = (
    <nav className="flex flex-col gap-2 p-4" aria-label="Haupt">
      {['Dashboard', 'Projekte', 'Dokumente', 'Workflows', 'Berichte', 'Einstellungen'].map((item) => (
        <a
          key={item}
          href="#"
          className="rounded px-2 py-1 text-sm hover:bg-muted focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
        >
          {item}
        </a>
      ))}
    </nav>
  );

  return (
    <div className="min-h-screen">
      {open && (
        <div className="fixed inset-0 z-40 md:hidden" role="dialog" aria-modal="true">
          <div className="absolute inset-0 bg-black/50" onClick={close} />
          <aside className="absolute left-0 top-0 h-full w-64 bg-bg shadow" data-testid="mobile-nav">
            {nav}
          </aside>
        </div>
      )}

      <header className="sticky top-0 z-30 border-b bg-bg">
        <div className="mx-auto flex h-14 w-full max-w-7xl items-center gap-2 px-4">
          <button
            ref={triggerRef}
            className="md:hidden rounded p-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            aria-label="Toggle navigation"
            onClick={() => setOpen(true)}
          >
            <span className="sr-only">Menu</span>
            <svg
              width="24"
              height="24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              aria-hidden="true"
            >
              <path strokeLinecap="round" strokeLinejoin="round" d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
          <input
            type="search"
            placeholder="Suche"
            className="flex-1 rounded border px-2 py-1 text-sm md:max-w-xs focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
          />
          <button
            className="rounded p-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            aria-label="Theme switch"
            onClick={() => document.documentElement.classList.toggle('dark')}
          >
            🌓
          </button>
          <button
            className="rounded p-2 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-accent"
            aria-label="User menu"
          >
            👤
          </button>
        </div>
      </header>

      <div className="mx-auto grid w-full max-w-7xl md:grid-cols-[240px_1fr]">
        <aside className="hidden border-r md:block">{nav}</aside>
        <div className="flex min-h-[calc(100vh-3.5rem)] flex-col">
          {breadcrumbs && (
            <div className="border-b px-4 py-2 text-sm" aria-label="Breadcrumb">
              {breadcrumbs}
            </div>
          )}
          {actions && <div className="flex items-center gap-2 px-4 py-2">{actions}</div>}
          <main className="flex-1 overflow-x-hidden px-4 py-2">{children}</main>
          <footer className="border-t px-4 py-2 text-xs text-muted">Build: {buildInfo ?? ''}</footer>
        </div>
      </div>
    </div>
  );
}

export default AppShell;
