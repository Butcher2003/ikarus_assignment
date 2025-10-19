import { Link, useLocation } from "react-router-dom";

const links = [
  { to: "/", label: "Recommendations" },
  { to: "/analytics", label: "Analytics" },
  { to: "/descriptions", label: "Descriptions" },
];

function Layout({ children }) {
  const location = useLocation();

  return (
    <div className="flex min-h-screen flex-col bg-slate-950 text-slate-100">
      <header className="border-b border-slate-800 bg-slate-900/70 backdrop-blur">
        <div className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
          <Link to="/" className="text-lg font-semibold text-primary-light">
            Furniture AI Assignment
          </Link>
          <nav className="flex items-center gap-4 text-sm font-medium">
            {links.map((link) => {
              const active = location.pathname === link.to;
              return (
                <Link
                  key={link.to}
                  to={link.to}
                  className={`rounded-md px-3 py-2 transition ${
                    active
                      ? "bg-primary/20 text-primary-light"
                      : "text-slate-300 hover:bg-slate-800"
                  }`}
                >
                  {link.label}
                </Link>
              );
            })}
          </nav>
        </div>
      </header>
      <main className="mx-auto flex w-full max-w-6xl flex-1 flex-col gap-6 px-6 py-8">
        {children}
      </main>
      <footer className="border-t border-slate-800 bg-slate-900/60">
        <div className="mx-auto flex max-w-6xl flex-wrap items-center justify-between gap-3 px-6 py-4 text-xs text-slate-500">
          <p>Powered by FastAPI, FAISS, and Tailwind</p>
          <p>Designed by : Aryan Adlakha</p>
          <p>&copy; {new Date().getFullYear()} Furniture AI Assignment</p>
        </div>
      </footer>
    </div>
  );
}

export default Layout;
