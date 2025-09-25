import { render } from "@testing-library/react";
import { axe } from "jest-axe";
import { StatusDot } from "./StatusDot";

describe("StatusDot", () => {
  it("applies correct classes", async () => {
    const { container, rerender } = render(<StatusDot status="success" />);
    expect(container.firstChild).toHaveClass("bg-accent");
    rerender(<StatusDot status="error" />);
    expect(container.firstChild).toHaveClass("bg-danger");
    rerender(<StatusDot status="warning" />);
    expect(container.firstChild).toHaveClass("bg-muted");
    rerender(<StatusDot status="neutral" />);
    expect(container.firstChild).toHaveClass("bg-fg");
    expect(await axe(container)).toHaveNoViolations();
  });
});
