import { render, screen } from "@testing-library/react";
import { axe } from "jest-axe";
import { Badge } from "./Badge";

describe("Badge", () => {
  it("renders text and is accessible", async () => {
    const { container } = render(<Badge>New</Badge>);
    expect(screen.getByText("New")).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});
