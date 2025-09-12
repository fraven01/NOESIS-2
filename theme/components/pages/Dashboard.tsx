import * as React from "react";
import { Card } from "../ui/Card";
import { Skeleton } from "../ui/Skeleton";
import { EmptyState } from "../ui/EmptyState";

/**
 * Static dashboard layout template with slots for future widgets.
 * @example
 * <Dashboard />
 */
export interface DashboardProps {
  KpiSlot?: React.ComponentType;
  ActivitySlot?: React.ComponentType;
  QuickLinksSlot?: React.ComponentType;
}

const DefaultKpiSlot: React.FC = () => (
  <div className="grid gap-4 sm:grid-cols-2 lg:grid-cols-4">
    {Array.from({ length: 4 }).map((_, index) => (
      <Card key={index}>
        <Skeleton className="h-24" />
      </Card>
    ))}
  </div>
);

const DefaultActivitySlot: React.FC = () => (
  <Card>
    <EmptyState title="Activity" text="Nothing here" />
  </Card>
);

const DefaultQuickLinksSlot: React.FC = () => (
  <Card>
    <EmptyState title="Quick Links" text="Nothing here" />
  </Card>
);

export const Dashboard: React.FC<DashboardProps> = ({
  KpiSlot = DefaultKpiSlot,
  ActivitySlot = DefaultActivitySlot,
  QuickLinksSlot = DefaultQuickLinksSlot,
}) => (
  <div className="flex flex-col gap-8">
    <section aria-label="KPI Area">
      <KpiSlot />
    </section>
    <div className="grid gap-8 lg:grid-cols-2">
      <section aria-label="Activity Area">
        <ActivitySlot />
      </section>
      <section aria-label="Quick Links">
        <QuickLinksSlot />
      </section>
    </div>
  </div>
);
