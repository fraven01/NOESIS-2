import { render } from "@testing-library/react";
import { axe } from "jest-axe";
import { Dashboard, DashboardProps } from "./Dashboard";

describe("Dashboard", () => {
  const props: DashboardProps = {
    kpis: [
      { label: "Users", value: "1.2k" },
      { label: "Sessions", value: "3.4k" },
      { label: "Errors", value: "23" },
      { label: "Uptime", value: "99.9%" },
    ],
    activities: [
      { id: "1", text: "User signed up", status: "success" },
      { id: "2", text: "Server restarted", status: "warning" },
    ],
    quickLinks: [
      { label: "Projects", href: "#" },
      { label: "Documents", href: "#" },
    ],
  };

  it("matches snapshot and is accessible", async () => {
    const { container } = render(<Dashboard {...props} />);
    expect(container).toMatchSnapshot();
    expect(await axe(container)).toHaveNoViolations();
  });
});
