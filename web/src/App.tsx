import { Route, Routes } from "react-router-dom";
import { Layout } from "./components/Layout";
import { ScenarioEditorPage } from "./pages/ScenarioEditorPage";
import { ScenarioListPage } from "./pages/ScenarioListPage";
import { SettingsPage } from "./pages/SettingsPage";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<ScenarioListPage />} />
        <Route path="/scenarios/:id" element={<ScenarioEditorPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Route>
    </Routes>
  );
}
