import { render, screen, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { axe } from "jest-axe";
import { useState, type FC } from "react";
import { DataTable, type Row } from "./DataTable";

describe("DataTable", () => {
  const rows: Row[] = [
    { id: "c", name: "Charlie" },
    { id: "a", name: "Alpha" },
    { id: "b", name: "Bravo" },
  ];

  const StatefulTable: FC = () => {
    const [selected, setSelected] = useState<string[]>([]);
    return <DataTable rows={rows} selected={selected} onSelectedChange={setSelected} />;
  };

  const getRowOrder = () =>
    screen
      .getAllByRole("row")
      .slice(1)
      .map((row) => within(row).getAllByRole("cell")[1].textContent ?? "");

  it("allows selecting rows via the header checkbox", async () => {
    const user = userEvent.setup();
    render(<StatefulTable />);

    const headerCheckbox = screen.getByRole("checkbox", { name: /select all/i });
    await user.click(headerCheckbox);

    let checkboxes = screen.getAllByRole("checkbox");
    for (const checkbox of checkboxes.slice(1)) {
      expect(checkbox).toBeChecked();
    }

    await user.click(headerCheckbox);

    checkboxes = screen.getAllByRole("checkbox");
    for (const checkbox of checkboxes.slice(1)) {
      expect(checkbox).not.toBeChecked();
    }
  });

  it("sorts rows when clicking the name column", async () => {
    const user = userEvent.setup();
    const { container } = render(<StatefulTable />);

    expect(getRowOrder()).toEqual(["Charlie", "Alpha", "Bravo"]);

    const nameHeader = screen.getByRole("columnheader", { name: "Name" });
    await user.click(nameHeader);
    expect(getRowOrder()).toEqual(["Alpha", "Bravo", "Charlie"]);

    await user.click(nameHeader);
    expect(getRowOrder()).toEqual(["Charlie", "Bravo", "Alpha"]);

    await user.click(nameHeader);
    expect(getRowOrder()).toEqual(["Charlie", "Alpha", "Bravo"]);

    expect(await axe(container)).toHaveNoViolations();
  });
});
