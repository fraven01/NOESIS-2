import { render } from "@testing-library/react";
import { axe } from "jest-axe";
import { Skeleton } from "./Skeleton";

describe("Skeleton", () => {
  it("renders and is accessible", async () => {
    const { container } = render(<Skeleton className="h-4 w-4" />);
    expect(container.firstChild).toBeInTheDocument();
    expect(await axe(container)).toHaveNoViolations();
  });
});
