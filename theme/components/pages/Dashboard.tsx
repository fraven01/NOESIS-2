import * as React from "react";
import { Card } from "../ui/Card";
import { StatusDot } from "../ui/StatusDot";
import { Skeleton } from "../ui/Skeleton";
import { EmptyState } from "../ui/EmptyState";

/**
 * Dashboard overview page with KPI cards, activity list and quick links.
 * @example
 * <Dashboard kpis={[{label:'Users', value:'1k'}]} activities={[]} quickLinks={[]} />
 */
export interface KPI {
  label: string;
  value: string;
}

export interface Activity {
  id: string;
  text: string;
  status: "success" | "warning" | "error" | "neutral";
}

export interface QuickLink {
  label: string;
  href: string;
}

export interface DashboardProps {
  kpis: KPI[];
  activities: Activity[];
  quickLinks: QuickLink[];
  loading?: boolean;
  error?: string;
}

export const Dashboard: React.FC<DashboardProps> = ({
  kpis,
  activities,
  quickLinks,
  loading = false,
  error,
}) => {
  if (error) {
    return <EmptyState title="Error" text={error} />;
  }

  return (
    <div className="flex flex-col gap-8">
      <section aria-label="KPI cards">
        <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
          {loading
            ? Array.from({ length: 4 }).map((_, i) => (
                <Skeleton key={i} className="h-24" />
              ))
            : kpis.slice(0, 4).map((kpi) => (
                <Card key={kpi.label} className="flex flex-col">
                  <span className="text-sm text-muted">{kpi.label}</span>
                  <span className="text-2xl font-semibold">{kpi.value}</span>
                </Card>
              ))}
        </div>
      </section>
      <section className="grid gap-8 lg:grid-cols-2">
        <Card>
          <h2 className="mb-4 text-lg font-semibold">Recent Activity</h2>
          {loading ? (
            <Skeleton className="h-32" />
          ) : activities.length ? (
            <ul className="space-y-4">
              {activities.map((activity) => (
                <li key={activity.id} className="flex items-center gap-2">
                  <StatusDot status={activity.status} />
                  <span className="text-sm text-fg">{activity.text}</span>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No activity" text="You're all caught up" />
          )}
        </Card>
        <Card>
          <h2 className="mb-4 text-lg font-semibold">Quick Links</h2>
          {loading ? (
            <Skeleton className="h-32" />
          ) : quickLinks.length ? (
            <ul className="space-y-2">
              {quickLinks.map((link) => (
                <li key={link.href + link.label}>
                  <a
                    href={link.href}
                    className="text-sm text-accent hover:underline focus:underline"
                  >
                    {link.label}
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState title="No links" text="Nothing here yet" />
          )}
        </Card>
      </section>
    </div>
  );
};

