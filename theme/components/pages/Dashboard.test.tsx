import { render } from "@testing-library/react";
import { axe } from "jest-axe";
import { Dashboard } from "./Dashboard";

describe("Dashboard", () => {
  it("renders regions and is accessible", async () => {
    const { getByRole, container } = render(<Dashboard />);
    expect(getByRole("region", { name: "KPI Area" })).toBeInTheDocument();
    expect(getByRole("region", { name: "Activity Area" })).toBeInTheDocument();
    expect(getByRole("region", { name: "Quick Links" })).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});
