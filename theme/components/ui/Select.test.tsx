import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { Select } from "./Select";

describe("Select", () => {
  it("changes value when selecting option", async () => {
    const user = userEvent.setup();
    const { container } = render(
      <Select aria-label="number" defaultValue="">
        <option value="">Choose</option>
        <option value="1">One</option>
      </Select>
    );
    const select = screen.getByRole("combobox");
    await user.selectOptions(select, "1");
    expect(select).toHaveValue("1");
    expect(await axe(container)).toHaveNoViolations();
  });
});
